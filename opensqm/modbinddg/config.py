"""Configuration for ModBinddG absolute binding free energy runs.

The defaults follow Sinko et al. (*PNAS* 2026) and its SI: a high-temperature
bound (ligand-protein) simulation and a 300 K unbound (ligand-in-solvent)
simulation, with a 0-2 Angstrom bound-state definition, a >=5 Angstrom
absorbing boundary, and 4 Angstrom cubic reweighting bins.
"""

from __future__ import annotations

import json
from typing import Literal

import xxhash
from openmm import unit
from pydantic import BaseModel, ConfigDict, Field
from pydantic_units import OpenMMQuantity

from opensqm.md.equilibrate import EquilibrationSettings

UnboundMode = Literal["explicit", "einstein"]
BoxShape = Literal["cube", "dodecahedron", "octahedron"]


def _default_equilibration_config() -> EquilibrationSettings:
    return EquilibrationSettings(
        npt_time=200 * unit.picoseconds,
        warmup_time=50 * unit.picoseconds,
    )


class ModBindDGSettings(BaseModel):
    """Settings for a single ModBinddG protein-ligand calculation."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # --- System preparation ---
    bound_box_shape: BoxShape = "cube"
    bound_padding: OpenMMQuantity[unit.nanometer] = 1.2 * unit.nanometer
    unbound_box_shape: BoxShape = "cube"
    unbound_padding: OpenMMQuantity[unit.nanometer] = 1.2 * unit.nanometer
    ionic_strength: float = 0.15

    # --- Equilibration ---
    cph_equilibration_ns: float = 1.0

    # --- Sampling ---
    # High temperature for the bound (unbinding) simulation; the unbound
    # ligand-only simulation always runs at ``unbound_temperature``.
    temperature: OpenMMQuantity[unit.kelvin] = 900.0 * unit.kelvin
    unbound_temperature: OpenMMQuantity[unit.kelvin] = 300.0 * unit.kelvin
    # Target temperature that populations are reweighted back to (Eq. 4).
    reference_temperature: OpenMMQuantity[unit.kelvin] = 300.0 * unit.kelvin
    integrator_step_size: OpenMMQuantity[unit.picosecond] = 0.002 * unit.picoseconds
    friction: OpenMMQuantity[unit.picosecond**-1] = 1.0 / unit.picosecond # type: ignore 

    n_replicas: int = 8
    # Optional per-replica bound temperatures (K). When omitted, every replica
    # uses ``temperature`` unless adaptive escape tuning is enabled.
    replica_temperatures: tuple[float, ...] | None = None
    # Target bound escape time (ns) for adaptive replica temperatures. When
    # set, replica 0 uses ``temperature`` and each subsequent replica is tuned
    # from the running ``ΔG°_well`` estimate and the previous escape time.
    ideal_escape_time_ns: float | None = None
    bound_frame_interval: OpenMMQuantity[unit.picosecond] = 10.0 * unit.picoseconds
    unbound_frame_interval: OpenMMQuantity[unit.picosecond] = 1.0 * unit.picoseconds
    # Upper bound on the length of any single escape trajectory.
    max_escape_time: OpenMMQuantity[unit.nanosecond] = 10.0 * unit.nanosecond
    # Target number of escape segments for the single long unbound simulation.
    unbound_target_escapes: int = 32
    unbound_max_time: OpenMMQuantity[unit.nanosecond] = 10.0 * unit.nanosecond

    # --- State definitions / reweighting (Angstrom) ---
    bound_state_radius: float = 2.0
    absorbing_boundary_radius: float = 5.0
    unbound_radius: float = 5.0
    bin_size: float = 4.0
    standard_volume: float = 1661.0  # Angstrom^3 (1 M standard state)
    pmf_max_radius: float = 30.0

    # --- Unbound free-energy estimation ---
    unbound_mode: UnboundMode = "explicit"
    # Stokes-Einstein-Sutherland diffusion estimate (SI Eq. SI-1..SI-4).
    diffusion_coefficient_m2_s: float = 0.5e-9
    einstein_radius: float = 5.0  # Angstrom

    # --- Statistics / output ---
    n_bootstrap: int = 1000
    random_seed: int = 1234
    n_closest_waters: int = 5

    @property
    def reweight_exponent(self) -> float:
        """Exponent for a single bound temperature (legacy convenience)."""
        t_sim = self.temperature.value_in_unit(unit.kelvin)
        t_ref = self.reference_temperature.value_in_unit(unit.kelvin)
        return t_sim / t_ref

    def bound_temperatures_K(self) -> tuple[float, ...]:
        """Simulation temperature (K) for each bound escape replica."""
        if self.replica_temperatures is not None:
            if len(self.replica_temperatures) != self.n_replicas:
                raise ValueError(
                    f"replica_temperatures has {len(self.replica_temperatures)} "
                    f"values but n_replicas={self.n_replicas}"
                )
            return self.replica_temperatures
        t_sim = self.temperature.value_in_unit(unit.kelvin)
        return (t_sim,) * self.n_replicas

    def bound_reweight_exponents(self) -> tuple[float, ...]:
        """Per-replica ``T_sim / T_ref`` exponents for bound-state reweighting."""
        t_ref = self.reference_temperature.value_in_unit(unit.kelvin)
        return tuple(t / t_ref for t in self.bound_temperatures_K())

    @property
    def unbound_volume(self) -> float:
        """Effective unbound state volume Vu (Angstrom^3): a sphere of radius."""
        import math

        return (4.0 / 3.0) * math.pi * self.unbound_radius**3

    def hash(self) -> str:
        """Reproducible short hash of the settings that define a run."""
        conf_dict = {
            "bound_box_shape": self.bound_box_shape,
            "bound_padding": str(self.bound_padding),
            "unbound_box_shape": self.unbound_box_shape,
            "unbound_padding": str(self.unbound_padding),
            "ionic_strength": self.ionic_strength,
            "temperature": str(self.temperature),
            "unbound_temperature": str(self.unbound_temperature),
            "reference_temperature": str(self.reference_temperature),
            "integrator_step_size": str(self.integrator_step_size),
            "n_replicas": self.n_replicas,
            "bound_frame_interval": str(self.bound_frame_interval),
            "unbound_frame_interval": str(self.unbound_frame_interval),
            "bound_state_radius": self.bound_state_radius,
            "absorbing_boundary_radius": self.absorbing_boundary_radius,
            "unbound_radius": self.unbound_radius,
            "bin_size": self.bin_size,
            "standard_volume": self.standard_volume,
            "unbound_mode": self.unbound_mode,
        }
        if self.replica_temperatures is not None:
            conf_dict["replica_temperatures"] = self.replica_temperatures
        if self.ideal_escape_time_ns is not None:
            conf_dict["ideal_escape_time_ns"] = self.ideal_escape_time_ns
        hash_str = json.dumps(conf_dict, sort_keys=True)
        return xxhash.xxh64(hash_str.encode()).hexdigest()
