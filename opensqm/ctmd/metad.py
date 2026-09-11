"""Well-tempered metadynamics on the ligand RMSD, and its c(t) bias offset.

The collective variable is the heavy-atom RMSD of the ligand from its docked
pose. PLUMED gets the protein-aligned RMSD by splitting the reference PDB into
alignment weights (protein) and displacement weights (ligand); OpenMM's
``RMSDForce`` aligns and measures the same atom set and cannot express that
split. It does not have to here: the bound state carries flat-bottom position
restraints on the backbone distal to the pocket, anchored to absolute reference
coordinates, so the protein cannot translate or rotate as a rigid body and the
lab-frame ligand RMSD already is the protein-aligned one. Loosening those
restraints silently breaks the CV.

c(t) is the time-dependent offset of Tiwary and Parrinello (*JPCB* 2015), what
PLUMED's ``METAD ... CALC_RCT`` prints as ``metad.rct``:

    c(t) = (1/beta) ln [ int ds e^{beta gamma/(gamma-1) V(s,t)}
                       / int ds e^{beta 1/(gamma-1) V(s,t)} ]

It is a functional of the whole bias surface at time ``t``, not of the bias the
walker felt, so it is computed from the grid ``Metadynamics`` already keeps
rather than logged along the trajectory. PLUMED refreshes it every
``RCT_USTRIDE`` hills; here it is recomputed every frame, which is strictly
more current and costs two 200-point reductions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from openmm import RMSDForce, unit
from openmm.app import DCDReporter
from openmm.app.metadynamics import BiasVariable, Metadynamics
from scipy.special import logsumexp
from tqdm import tqdm

from opensqm.modbind.escape import build_simulation

if TYPE_CHECKING:
    from opensqm.ctmd.config import CTMDSettings
    from opensqm.modbind.states import PreparedState


@dataclass
class CTMDTrajectory:
    """One replica's collective variable and bias offset, sampled every frame."""

    rmsd_nm: np.ndarray
    ct_kj: np.ndarray
    frame_interval_ps: float

    @property
    def duration_ns(self) -> float:
        """Simulated time of this replica."""
        return (len(self.rmsd_nm) - 1) * self.frame_interval_ps / 1000.0


def ct_from_bias(bias_kj: np.ndarray, kt_kj: float, bias_factor: float) -> float:
    """c(t) in kJ/mol for a bias ``V(s,t)`` tabulated on a uniform grid.

    ``bias_kj`` is V(s,t) in kJ/mol over the collective-variable grid, ``kt_kj``
    is kT in kJ/mol and ``bias_factor`` is the well-tempered gamma. The grid
    spacing cancels in the ratio, so only the values matter. A uniform bias of
    height V returns exactly V.
    """
    beta_v = np.ravel(np.asarray(bias_kj, dtype=np.float64)) / kt_kj
    scale = bias_factor / (bias_factor - 1.0)
    return kt_kj * float(logsumexp(beta_v * scale) - logsumexp(beta_v / (bias_factor - 1.0)))


def ct(meta: Metadynamics) -> float:
    """c(t) of a running metadynamics, in kJ/mol. Equals PLUMED's ``metad.rct``."""
    gamma = meta.biasFactor
    kt_kj = (unit.MOLAR_GAS_CONSTANT_R * meta.temperature).value_in_unit(unit.kilojoule_per_mole)
    # getFreeEnergy() returns -(gamma/(gamma-1)) V(s,t); invert it to recover V.
    free_energy = meta.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole)
    return ct_from_bias(-free_energy * (gamma - 1.0) / gamma, kt_kj, gamma)


def build_metadynamics(
    state: PreparedState, config: CTMDSettings
) -> tuple[PreparedState, Metadynamics]:
    """Add the RMSD bias to a copy of ``state``'s system.

    Returns the copied state and the ``Metadynamics`` driving it. The copy
    matters: ``Metadynamics.__init__`` calls ``System.addForce``, and the
    prepared state's system is cached to XML and reused across replicas, so
    biasing it in place would poison that cache.
    """
    system = copy.deepcopy(state.system)
    # RMSDForce needs reference coordinates for every particle even though only
    # the ligand heavy atoms (already the contents of ligand_indices) are scored.
    variable = BiasVariable(
        RMSDForce(state.positions, state.ligand_indices),
        minValue=0.0,
        maxValue=config.grid_max.value_in_unit(unit.nanometer),
        biasWidth=config.hill_sigma.value_in_unit(unit.nanometer),
        gridWidth=config.grid_bins,
    )
    meta = Metadynamics(
        system,
        [variable],
        config.temperature,
        config.bias_factor,
        config.hill_height,
        config.hill_interval_steps,
    )
    return replace(state, system=system), meta


def run_ctmd_replica(
    state: PreparedState,
    config: CTMDSettings,
    *,
    seed: int,
    dcd_path: str | None = None,
) -> CTMDTrajectory:
    """Run one metadynamics replica until the ligand commits to leaving.

    Records the RMSD and c(t) every ``frame_interval``, starting from the
    unbiased frame at t = 0, and stops early once the RMSD has held above
    ``commit_cutoff`` for a full ``commit_time`` window or ``max_time`` is
    reached. The scored frame is re-derived from ``rmsd_nm`` by
    :func:`opensqm.ctmd.score.commit_frame`, so the counter below only ends the
    run; it never sets the score.
    """
    biased_state, meta = build_metadynamics(state, config)
    simulation = build_simulation(
        biased_state,
        temperature=config.temperature,
        step_size=config.integrator_step_size,
        friction=config.friction,
        seed=seed,
    )
    if dcd_path is not None:
        simulation.reporters.append(DCDReporter(dcd_path, config.frame_interval_steps))

    cutoff = config.commit_cutoff.value_in_unit(unit.nanometer)
    rmsds = [float(meta.getCollectiveVariables(simulation)[0])]
    cts = [ct(meta)]
    above = 0
    for _ in tqdm(range(config.max_frames), desc="CTMD replica", unit="frame"):
        meta.step(simulation, config.frame_interval_steps)
        rmsd = float(meta.getCollectiveVariables(simulation)[0])
        rmsds.append(rmsd)
        cts.append(ct(meta))
        above = above + 1 if rmsd > cutoff else 0
        if above >= config.commit_frames:
            break

    trajectory = CTMDTrajectory(
        rmsd_nm=np.asarray(rmsds, dtype=np.float64),
        ct_kj=np.asarray(cts, dtype=np.float64),
        frame_interval_ps=config.frame_interval.value_in_unit(unit.picosecond),
    )
    logger.info(
        f"Replica ran {trajectory.duration_ns:.3f} ns, "
        f"final RMSD {rmsds[-1]:.2f} nm, c(t) {cts[-1]:.1f} kJ/mol"
    )
    return trajectory
