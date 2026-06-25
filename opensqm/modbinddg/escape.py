"""High-temperature escape simulations for ModBinddG.

Each escape simulation tracks the displacement of the ligand centre of mass
(COM) from its starting position, recording a frame every
``frame_interval``. The bound state runs as independent replicas that stop
when the COM first crosses the absorbing boundary. The unbound state runs as a
single long trajectory whose origin is reset every time the COM crosses the
boundary, mimicking many independent escape segments (SI: "each time the ligand
crossed the 5 A threshold, the origin of the ligand was reset").

For the bound state, displacements are measured *referenced to the protein*
(SI: "ligand positional movements using COM Cartesian coordinates from the
starting position referenced to the protein structure"). Each frame is rigidly
superposed onto the equilibrated protein via Kabsch alignment of the backbone
C-alpha atoms, the ligand COM is mapped through that transform, and its
displacement is taken from the reference (equilibrated) ligand COM. This removes
protein rigid-body translation/rotation so the coordinate reflects only ligand
motion relative to the pocket. The unbound state has no protein, so its COM is
tracked directly in the lab frame against a fixed origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mdtraj as md
import numpy as np
from loguru import logger
from openmm import LangevinMiddleIntegrator, unit
from openmm.app import DCDReporter, Simulation
from tqdm import tqdm

if TYPE_CHECKING:
    from opensqm.modbinddg.config import ModBindDGSettings
    from opensqm.modbinddg.states import PreparedState

_NM_TO_ANGSTROM = 10.0

_NON_PROTEIN_RESIDUES = {
    "LIG", "COF", "HOH", "SOL", "WAT", "NA", "CL", "MG", "K", "ZN", "ACE", "NME",
}


@dataclass
class BoundEscapeDiagnostics:
    """Per-replica diagnostics for a bound escape run."""

    escape_time_ns: float
    escaped: bool
    calpha_rmsd_last10_A: float


def _calpha_indices(topology) -> list[int]:
    return [
        atom.index
        for atom in topology.atoms()
        if atom.name == "CA" and atom.residue.name not in _NON_PROTEIN_RESIDUES
    ]


def _ligand_masses(state: PreparedState) -> np.ndarray:
    return np.array(
        [
            state.system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in state.ligand_indices
        ],
        dtype=np.float64,
    )


def _kabsch_rt(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rigid transform mapping ``mobile`` onto ``target`` (Kabsch).

    Returns ``(R, t)`` such that ``x_aligned = R @ x + t`` best superposes the
    ``mobile`` point set onto ``target`` in a least-squares sense. The
    reflection-correcting determinant sign keeps ``R`` a proper rotation.
    """
    mob_c = mobile.mean(axis=0)
    tgt_c = target.mean(axis=0)
    p = mobile - mob_c
    q = target - tgt_c
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    translation = tgt_c - rotation @ mob_c
    return rotation, translation


def _ligand_com_nm(simulation: Simulation, ligand_indices: list[int], masses: np.ndarray) -> np.ndarray:
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    ligand_positions = positions[ligand_indices]
    return np.average(ligand_positions, axis=0, weights=masses)


def _build_simulation(
    state: PreparedState,
    *,
    temperature: unit.Quantity,
    step_size: unit.Quantity,
    friction: unit.Quantity,
    seed: int,
) -> Simulation:
    integrator = LangevinMiddleIntegrator(temperature, friction, step_size)
    if seed:
        integrator.setRandomNumberSeed(int(seed))
    simulation = Simulation(state.topology, state.system, integrator)
    box_vectors = state.topology.getPeriodicBoxVectors()
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(state.positions)
    simulation.context.setVelocitiesToTemperature(temperature, int(seed) or None)
    return simulation


def run_bound_escape(
    state: PreparedState,
    config: ModBindDGSettings,
    *,
    seed: int,
    dcd_path: str | None = None,
    temperature: unit.Quantity | None = None,
) -> tuple[np.ndarray, BoundEscapeDiagnostics]:
    """Run one bound-state replica until the ligand COM escapes.

    Returns ``(displacements, diagnostics)`` where ``displacements`` is a
    ``(n_frames, 3)`` array of COM displacements (Angstrom) from the starting
    position (first row at the origin), and ``diagnostics`` carries the escape
    time and the backbone C-alpha RMSD of the final 10% of frames measured
    against the equilibrated protein configuration (best-fit superposition via
    :func:`mdtraj.rmsd`).
    """
    logger.info(f"Running bound escape at {temperature._value:.0f} K")
    frame_interval_steps = max(
        1, round(config.bound_frame_interval / config.integrator_step_size)
    )
    max_frames = max(
        1, round(config.max_escape_time / config.bound_frame_interval)
    )

    simulation = _build_simulation(
        state,
        temperature=temperature or config.temperature,
        step_size=config.integrator_step_size,
        friction=config.friction,
        seed=seed,
    )
    if dcd_path is not None:
        simulation.reporters.append(DCDReporter(dcd_path, frame_interval_steps))

    masses = _ligand_masses(state)
    calpha_indices = _calpha_indices(state.topology)
    ref_positions = np.asarray(
        state.positions.value_in_unit(unit.nanometer), dtype=np.float64
    )
    calpha_topology = (
        md.Topology.from_openmm(state.topology).subset(calpha_indices)
        if calpha_indices
        else None
    )
    reference_calpha = (
        md.Trajectory(ref_positions[calpha_indices][None], calpha_topology)
        if calpha_indices
        else None
    )
    # Reference geometry for protein-frame displacement (paper: ligand COM
    # referenced to the protein structure). Falls back to lab frame if the
    # system has no C-alpha atoms (e.g. a protein-free system).
    ref_calpha_xyz = ref_positions[calpha_indices] if calpha_indices else None
    ref_ligand_com = np.average(
        ref_positions[state.ligand_indices], axis=0, weights=masses
    )

    def _positions_nm() -> np.ndarray:
        snapshot = simulation.context.getState(
            getPositions=True, enforcePeriodicBox=False
        )
        return snapshot.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    def _displacement_A(positions: np.ndarray) -> np.ndarray:
        com = np.average(positions[state.ligand_indices], axis=0, weights=masses)
        if ref_calpha_xyz is not None:
            rotation, translation = _kabsch_rt(positions[calpha_indices], ref_calpha_xyz)
            com = rotation @ com + translation
        return (com - ref_ligand_com) * _NM_TO_ANGSTROM

    displacements: list[np.ndarray] = [_displacement_A(_positions_nm())]
    calpha_frames: list[np.ndarray] = []
    escaped = False
    for _ in tqdm(range(max_frames), desc="Bound escape", unit="frame"):
        simulation.step(frame_interval_steps)
        positions = _positions_nm()
        displacement = _displacement_A(positions)
        displacements.append(displacement)
        if calpha_topology is not None:
            calpha_frames.append(positions[calpha_indices])
        if np.linalg.norm(displacement) >= config.absorbing_boundary_radius:
            escaped = True
            break

    n_steps = len(displacements) - 1
    escape_time_ns = n_steps * config.bound_frame_interval.value_in_unit(unit.nanosecond)

    rmsd = float("nan")
    if calpha_frames and reference_calpha is not None:
        n_last = max(1, math.ceil(0.10 * len(calpha_frames)))
        last = md.Trajectory(
            np.asarray(calpha_frames[-n_last:], dtype=np.float32), calpha_topology
        )
        rmsd = float(md.rmsd(last, reference_calpha, 0).mean()) * _NM_TO_ANGSTROM

    diagnostics = BoundEscapeDiagnostics(
        escape_time_ns=escape_time_ns, escaped=escaped, calpha_rmsd_last10_A=rmsd
    )
    return np.asarray(displacements, dtype=np.float64), diagnostics


def run_unbound_escape(
    state: PreparedState,
    config: ModBindDGSettings,
) -> tuple[list[np.ndarray], np.ndarray, bool]:
    """Run the single long unbound trajectory, resetting at each escape.

    Returns ``(segments, escape_times_steps, converged)`` where each segment is
    a ``(n_frames, 3)`` array of COM displacements (Angstrom) from the segment
    origin, and ``escape_times_steps`` is the number of MD steps in each
    completed escape segment.
    """
    frame_interval_steps = max(
        1, round(config.unbound_frame_interval / config.integrator_step_size)
    )
    max_total_frames = max(
        1, round(config.unbound_max_time / config.unbound_frame_interval)
    )
    max_segment_frames = max(
        1, round(config.max_escape_time / config.unbound_frame_interval)
    )

    simulation = _build_simulation(
        state,
        temperature=config.unbound_temperature,
        step_size=config.integrator_step_size,
        friction=config.friction,
        seed=config.random_seed,
    )

    masses = _ligand_masses(state)
    origin = _ligand_com_nm(simulation, state.ligand_indices, masses)

    segments: list[np.ndarray] = []
    escape_times_steps: list[int] = []

    current: list[np.ndarray] = [np.zeros(3)]
    segment_frames = 0
    total_frames = 0

    with tqdm(total=max_total_frames, desc="Unbound sampling", unit="frame") as pbar:
        while (
            total_frames < max_total_frames
            and len(segments) < config.unbound_target_escapes
        ):
            simulation.step(frame_interval_steps)
            total_frames += 1
            segment_frames += 1
            pbar.update(1)
            com = _ligand_com_nm(simulation, state.ligand_indices, masses)
            displacement = (com - origin) * _NM_TO_ANGSTROM
            current.append(displacement)

            escaped = np.linalg.norm(displacement) >= config.absorbing_boundary_radius
            if escaped or segment_frames >= max_segment_frames:
                if escaped:
                    segments.append(np.asarray(current, dtype=np.float64))
                    escape_times_steps.append(segment_frames * frame_interval_steps)
                origin = com
                current = [np.zeros(3)]
                segment_frames = 0
            pbar.set_postfix(escapes=f"{len(segments)}/{config.unbound_target_escapes}")

    converged = len(segments) >= config.unbound_target_escapes
    if not converged:
        logger.warning(
            f"Unbound simulation recorded {len(segments)} escapes "
            f"(target {config.unbound_target_escapes}) before time limit"
        )

    return segments, np.asarray(escape_times_steps, dtype=np.int64), converged
