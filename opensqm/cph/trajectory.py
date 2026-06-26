"""Per-system-state DCD trajectory manager for ConstantPH simulations.

A "system state" is the joint tuple ``(currentIndex, ...)`` over the
ConstantPH titratable residues (in sorted residue-index order). Because
``ResidueState`` swap is in-place on a single simulation context, the
*atom count* of the running system never changes - only the live
``NonbondedForce`` parameters of a small set of atoms do (the
"titratable" hydrogens, which get zero charge in variants where they
are absent, plus the two water-swap ghost slots when WaterSwap is
enabled). This module reaps that fact to write one DCD per state with
the *physically present* atoms only:

* The water-swap ``ghost2`` slot (the off slot) is always excluded -
  between attempts it carries zero charge and zero epsilon so it is
  not a physical water. ``ghost1`` (the on slot) is *kept* because it
  is fully coupled at every reporter emit time; mid-NCMC frames where
  ghost1 is part-decoupled are dropped via the ``lambda_water_swap``
  guard rather than the keep mask.
* For every titratable residue, hydrogens whose live NonbondedForce
  charge is ``0.0`` in the current state are excluded (ConstantPH
  zeroes the charge of variant-absent hydrogens via
  ``_get_zero_parameters``). The candidate set comes from
  ``titration.explicitHydrogenIndices`` so we don't scan all atoms.

Per-state caching is one-shot: the first time a given state combo is
encountered we compute its keep-mask, build a stripped sub-topology,
write a matching ``.pdb``, and open a fresh ``.dcd``. Subsequent visits
to the same state reuse the cached writer.

A sidecar CSV (``<base>.csv`` by default) is written with one row per
emitted frame:

    step,time_ns,system_state,frame_ix

where ``frame_ix`` is the 0-based frame index inside *that state's*
DCD (so re-reading the CSV is enough to align the global trajectory
with the per-state DCDs).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mdtraj as md
import numpy as np
from openmm import NonbondedForce, unit
from openmm.app import Modeller, PDBFile
from openmm.app.dcdfile import DCDFile

if TYPE_CHECKING:
    from opensqm.cph.constantph import ConstantPH


class _StateChannel:
    """Bookkeeping for one system-state's DCD + frame counter."""

    __slots__ = (
        "dcd_file",
        "dcd_path",
        "frame_ix",
        "keep_indices",
        "out_handle",
        "pdb_path",
    )

    def __init__(
        self,
        dcd_path: Path,
        pdb_path: Path,
        keep_indices: np.ndarray,
        dcd_file: DCDFile,
        out_handle: Any,
        *,
        frame_ix: int = 0,
    ) -> None:
        self.dcd_path = dcd_path
        self.pdb_path = pdb_path
        self.keep_indices = keep_indices
        self.dcd_file = dcd_file
        self.out_handle = out_handle
        self.frame_ix = frame_ix

    def close(self) -> None:
        try:
            self.out_handle.close()
        except Exception:
            pass


class StateSplitTrajectoryManager:
    """OpenMM reporter that writes one DCD per ConstantPH system state.

    Plug into ``cph.simulation.reporters`` directly:

        manager = StateSplitTrajectoryManager(cph, "out/traj", report_interval=500)
        cph.simulation.reporters.append(manager)
        ...
        manager.close()  # flush DCD/CSV at the end of the run

    Parameters
    ----------
    cph : ConstantPH
        The driving constant-pH model. Used to read titration state,
        the explicit topology, and the optional water-swap ghost slots.
    base_path : str | Path
        Base path used to derive output filenames. For each unique
        state tuple ``(s0, s1, ...)`` encountered the manager writes
        ``<base>.<s0>_<s1>_...>.dcd`` and a matching ``.pdb`` of the
        stripped topology.
    report_interval : int
        Number of integration steps between successive reports. The
        OpenMM reporter protocol uses this to compute the number of
        steps until the next call to :meth:`report`.
    csv_path : str | Path, optional
        Override the default sidecar CSV path. Defaults to
        ``<base>.csv``.
    enforce_periodic_box : bool, optional
        If True (the default), positions are wrapped into the periodic
        box before being written. Forwarded to OpenMM's reporter API
        via :meth:`describeNextReport`.
    """

    def __init__(
        self,
        cph: "ConstantPH",
        base_path: str | Path,
        report_interval: int,
        *,
        csv_path: str | Path | None = None,
        enforce_periodic_box: bool = True,
        append: bool = False,
    ) -> None:
        self.cph = cph
        self.simulation = cph.simulation
        self.system = self.simulation.system
        self.topology = cph.explicitTopology
        self.report_interval = int(report_interval)
        self.enforce_periodic_box = bool(enforce_periodic_box)
        self.append = bool(append)

        self.base_path = Path(base_path)
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = (
            Path(csv_path) if csv_path is not None
            else self.base_path.with_suffix(".csv")
        )

        self._nb_force = self._find_nonbonded_force()
        self._titration_keys = sorted(self.cph.titrations)
        self._titratable_h_indices = np.asarray(
            sorted({
                int(i)
                for k in self._titration_keys
                for i in self.cph.titrations[k].explicitHydrogenIndices
            }),
            dtype=int,
        )

        # Only ghost2 is excluded from the trajectory. Between attempts
        # ghost2 is the off slot (zero charge and zero epsilon) and
        # therefore not a physical water; ghost1 is the on slot,
        # carrying full TIP3P parameters at every reporter emit time
        # (the NCMC-mid-switch frames where ghost1 is part-decoupled
        # are dropped by the ``lambda_water_swap != 0`` guard in
        # ``report``). Label swap on acceptance simply means a
        # different physical water has rotated into the atom indices
        # tagged ``ghost1`` - that's no different from any other water
        # diffusing through space, so we keep ghost1 and drop only
        # ghost2.
        self._always_ghost: frozenset[int] = frozenset()

        self._channels: dict[tuple[int, ...], _StateChannel] = {}
        self._initial_frame_ix = self._restore_frame_counts_from_csv()
        self._step_size_ps = float(
            self.simulation.integrator.getStepSize()
            .value_in_unit(unit.picosecond)
        )

        resume_csv = self.append and self.csv_path.exists()
        self._csv_handle = open(
            self.csv_path,
            "a" if resume_csv else "w",
            newline="",
        )
        self._csv_writer = csv.writer(self._csv_handle)
        if not resume_csv:
            self._csv_writer.writerow(["step", "time_ns", "system_state", "frame_ix"])
            self._csv_handle.flush()

    def _find_nonbonded_force(self) -> NonbondedForce:
        """Locate the single :class:`NonbondedForce` on the explicit system."""
        candidates = [
            f for f in self.system.getForces()
            if isinstance(f, NonbondedForce)
        ]
        if not candidates:
            raise RuntimeError(
                "StateSplitTrajectoryManager requires a NonbondedForce "
                "on the ConstantPH explicit system"
            )
        if len(candidates) > 1:
            raise RuntimeError(
                "StateSplitTrajectoryManager found multiple NonbondedForce "
                "instances; an unambiguous force is required"
            )
        return candidates[0]

    def _restore_frame_counts_from_csv(self) -> dict[tuple[int, ...], int]:
        """Return next frame index per system-state signature from an existing CSV."""
        if not self.csv_path.exists():
            return {}
        counts: dict[tuple[int, ...], int] = {}
        with open(self.csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                signature = tuple(int(x) for x in row["system_state"].split("_"))
                counts[signature] = int(row["frame_ix"]) + 1
        return counts

    # ------------------------------------------------------------------
    # OpenMM reporter protocol
    # ------------------------------------------------------------------

    def describeNextReport(self, simulation):  # noqa: N802 - OpenMM API
        steps = (
            self.report_interval
            - simulation.currentStep % self.report_interval
        )
        # (steps, positions, velocities, forces, energy, enforcePeriodicBox)
        return (steps, True, False, False, False, self.enforce_periodic_box)

    def report(self, simulation, state) -> None:
        # Skip frames emitted mid-NCMC water-swap. ``WaterSwapMC.attempt``
        # drives ``simulation.step()`` between alchemy jumps, so the
        # production stepper can land on a reporter interval while
        # ``lambda_water_swap > 0`` - the Hamiltonian at that moment is
        # the mid-switch alchemical interpolation, not the physical one,
        # and emitting the frame would pollute the per-state DCD.


        signature = self._current_signature()
        channel = self._channels.get(signature)
        if channel is None:
            channel = self._create_channel(signature)
            self._channels[signature] = channel

        positions_nm = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=float,
        )
        filtered = positions_nm[channel.keep_indices] * unit.nanometer
        channel.dcd_file.writeModel(
            filtered, periodicBoxVectors=state.getPeriodicBoxVectors(),
        )
        channel.out_handle.flush()

        time_ns = simulation.currentStep * self._step_size_ps * 1e-3
        sig_str = "_".join(str(s) for s in signature)
        self._csv_writer.writerow([
            simulation.currentStep,
            f"{time_ns:.6f}",
            sig_str,
            channel.frame_ix,
        ])
        self._csv_handle.flush()
        channel.frame_ix += 1

    # ------------------------------------------------------------------
    # State-channel construction
    # ------------------------------------------------------------------

    def _current_signature(self) -> tuple[int, ...]:
        return tuple(
            int(self.cph.titrations[k].currentIndex)
            for k in self._titration_keys
        )

    def _compute_keep_indices(self) -> np.ndarray:
        """Atom indices physically present in the current ConstantPH state.

        Excludes:
        * water-swap ghost slot atoms (precomputed in
          :attr:`_always_ghost`),
        * titratable hydrogens whose live NonbondedForce charge is 0
          (i.e. zeroed by ConstantPH for the current variant).

        Computed at most once per unique state combination - the
        caller caches the resulting mask on the per-state
        :class:`_StateChannel`.
        """
        dead_h: set[int] = set()
        for atom_idx in self._titratable_h_indices:
            charge, _sigma, _epsilon = self._nb_force.getParticleParameters(
                int(atom_idx),
            )
            if float(charge.value_in_unit(unit.elementary_charge)) == 0.0:
                dead_h.add(int(atom_idx))
        excluded = self._always_ghost | dead_h
        n = self.system.getNumParticles()
        return np.asarray(
            [i for i in range(n) if i not in excluded],
            dtype=int,
        )

    def _create_channel(self, signature: tuple[int, ...]) -> _StateChannel:
        keep = self._compute_keep_indices()
        sig_str = "_".join(str(s) for s in signature)
        dcd_path = self.base_path.with_name(
            f"{self.base_path.name}.{sig_str}.dcd"
        )
        pdb_path = self.base_path.with_name(
            f"{self.base_path.name}.{sig_str}.pdb"
        )

        append_dcd = self.append and dcd_path.exists()
        if append_dcd:
            sub_topology = PDBFile(str(pdb_path)).topology
        else:
            sub_topology, sub_positions = self._build_sub_topology(keep)
            with open(pdb_path, "w") as f:
                PDBFile.writeFile(sub_topology, sub_positions, f)

        mode = "r+b" if append_dcd else "wb"
        out_handle = open(dcd_path, mode)
        dcd = DCDFile(
            out_handle,
            sub_topology,
            self.simulation.integrator.getStepSize(),
            self.simulation.currentStep,
            self.report_interval,
            append=append_dcd,
        )
        return _StateChannel(
            dcd_path=dcd_path,
            pdb_path=pdb_path,
            keep_indices=keep,
            dcd_file=dcd,
            out_handle=out_handle,
            frame_ix=self._initial_frame_ix.get(signature, 0),
        )

    def _build_sub_topology(self, keep_indices: np.ndarray):
        """Return ``(topology, positions)`` containing only kept atoms."""
        positions = (
            self.simulation.context.getState(getPositions=True)
            .getPositions()
        )
        modeller = Modeller(self.topology, positions)
        keep_set = {int(i) for i in keep_indices}
        atoms = list(self.topology.atoms())
        to_delete = [a for a in atoms if a.index not in keep_set]
        modeller.delete(to_delete)
        return modeller.topology, modeller.positions

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close every per-state DCD plus the CSV sidecar."""
        for ch in self._channels.values():
            ch.close()
        try:
            self._csv_handle.close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


TRAJECTORIES_DIRNAME = "trajectories"


def trajectories_dir(output_path: Path) -> Path:
    return output_path / TRAJECTORIES_DIRNAME


def replica_trajectory_stem(replica_i: int) -> str:
    return f"replica_{replica_i:04d}"


def replica_trajectory_base(output_path: Path, replica_i: int) -> Path:
    return trajectories_dir(output_path) / replica_trajectory_stem(replica_i)


def replica_trajectory_index(output_path: Path, replica_i: int) -> Path:
    return replica_trajectory_base(output_path, replica_i).with_suffix(".csv")


def image_dcd_inplace(dcd_path: Path, pdb_path: Path) -> None:
    """Wrap molecules into the primary unit cell, overwriting ``dcd_path``."""
    traj = md.load_dcd(str(dcd_path), top=str(pdb_path))
    if traj.n_frames == 0:
        return
    traj.image_molecules(inplace=True)
    traj.save_dcd(str(dcd_path))


def iter_replica_state_trajectories(
    output_path: Path,
    replica_i: int,
) -> list[tuple[Path, Path]]:
    """Return (dcd, pdb) pairs for one replica's state-split trajectories.

    Each DCD is molecule-imaged in place before being returned.
    """
    traj_dir = trajectories_dir(output_path)
    prefix = replica_trajectory_stem(replica_i)
    pairs: list[tuple[Path, Path]] = []
    for dcd in sorted(traj_dir.glob(f"{prefix}.*.dcd")):
        if dcd.name.endswith(".close.dcd") or dcd.name.endswith("_imaged.dcd"):
            continue
        pdb = dcd.with_suffix(".pdb")
        image_dcd_inplace(dcd, pdb)
        pairs.append((dcd, pdb))
    return pairs
