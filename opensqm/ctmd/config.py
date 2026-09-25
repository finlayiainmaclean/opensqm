"""Configuration for CTMD hit triage.

Defaults reproduce Shekhar et al. (bioRxiv 2026, doi:10.64898/2026.02.05.703972)
and its reference PLUMED input: well-tempered metadynamics on the ligand RMSD
from the docked pose, 1.75 kJ/mol hills every 1 ps with sigma 0.015 nm and bias
factor 10, 5 ns per replica, and a ligand called unbound once its RMSD holds
above 6 A for 200 ps.
"""

from __future__ import annotations

import json

import xxhash
from openmm import unit
from pydantic import BaseModel, ConfigDict
from pydantic_units import OpenMMQuantity


class CTMDSettings(BaseModel):
    """Settings for one CTMD run on a single protein-ligand pair.

    Construct with no arguments to get the paper's protocol. The reported score
    is the minimum c(t) over ``n_score_replicas`` replicas drawn from the
    ``n_replicas`` collected, so ``n_replicas`` must be at least
    ``n_score_replicas``; collecting more than you score is what gives the
    bootstrap its spread (the paper collects 10 and scores 3).

    A higher c(t) means more reversible work was needed to push the ligand out
    of its pose, so it ranks as the better binder.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # --- Equilibration ---
    # Production MD length (ns) of the MMGBSA protomer funnel that picks the
    # ligand protonation state and the starting pose for the replicas.
    mmgbsa_equilibration_ns: float = 0.1

    # --- Sampling ---
    temperature: OpenMMQuantity[unit.kelvin] = 300.0 * unit.kelvin
    integrator_step_size: OpenMMQuantity[unit.picosecond] = 0.002 * unit.picoseconds
    friction: OpenMMQuantity[unit.picosecond**-1] = 1.0 / unit.picosecond  # type: ignore
    n_replicas: int = 10
    n_score_replicas: int = 3
    frame_interval: OpenMMQuantity[unit.picosecond] = 0.2 * unit.picoseconds
    max_time: OpenMMQuantity[unit.nanosecond] = 5.0 * unit.nanosecond

    # --- Well-tempered metadynamics ---
    hill_height: OpenMMQuantity[unit.kilojoule_per_mole] = 1.75 * unit.kilojoule_per_mole
    hill_sigma: OpenMMQuantity[unit.nanometer] = 0.015 * unit.nanometer
    hill_interval: OpenMMQuantity[unit.picosecond] = 1.0 * unit.picoseconds
    bias_factor: float = 10.0
    # RMSD grid, from 0 to ``grid_max``. Beyond it OpenMM sets the bias force to
    # zero (PLUMED instead aborts the run), which is harmless here: the ligand is
    # already committed to leaving long before the RMSD reaches 1.5 nm.
    grid_max: OpenMMQuantity[unit.nanometer] = 1.5 * unit.nanometer
    grid_bins: int = 200

    # --- Scoring ---
    commit_cutoff: OpenMMQuantity[unit.nanometer] = 0.6 * unit.nanometer
    commit_time: OpenMMQuantity[unit.picosecond] = 200.0 * unit.picoseconds
    # Two ligands whose c(t) differ by less than this (1 kT) are tied, and the
    # longer-lived one ranks first.
    tie_tolerance: OpenMMQuantity[unit.kilojoule_per_mole] = 2.5 * unit.kilojoule_per_mole
    n_bootstrap: int = 250
    random_seed: int = 1234

    @property
    def frame_interval_steps(self) -> int:
        """MD steps between logged frames."""
        return max(1, round(self.frame_interval / self.integrator_step_size))

    @property
    def hill_interval_steps(self) -> int:
        """MD steps between deposited hills (PLUMED ``PACE``)."""
        return max(1, round(self.hill_interval / self.integrator_step_size))

    @property
    def max_frames(self) -> int:
        """Frame cap for one replica."""
        return max(1, round(self.max_time / self.frame_interval))

    @property
    def commit_frames(self) -> int:
        """Consecutive frames above the cutoff that count as unbound."""
        return max(1, round(self.commit_time / self.frame_interval))

    def hash(self) -> str:
        """Reproducible short hash of the settings that define a run."""
        conf_dict = {
            "temperature": str(self.temperature),
            "integrator_step_size": str(self.integrator_step_size),
            "n_replicas": self.n_replicas,
            "n_score_replicas": self.n_score_replicas,
            "frame_interval": str(self.frame_interval),
            "max_time": str(self.max_time),
            "hill_height": str(self.hill_height),
            "hill_sigma": str(self.hill_sigma),
            "hill_interval": str(self.hill_interval),
            "bias_factor": self.bias_factor,
            "grid_max": str(self.grid_max),
            "grid_bins": self.grid_bins,
            "commit_cutoff": str(self.commit_cutoff),
            "commit_time": str(self.commit_time),
        }
        return xxhash.xxh64(json.dumps(conf_dict, sort_keys=True).encode()).hexdigest()
