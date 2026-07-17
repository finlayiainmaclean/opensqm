"""Configuration for ModBinddG absolute binding free energy runs.

The defaults follow Sinko et al. (*PNAS* 2026) and its SI: high-temperature
bound (ligand-protein) and unbound (ligand-in-solvent) simulations run at the
*same* temperature, with a 0-2 Angstrom bound-state definition, a >=5 Angstrom
absorbing boundary, and 4 Angstrom cubic reweighting bins.
"""

from __future__ import annotations

import json
from typing import Literal

import xxhash
from openmm import unit
from pydantic import BaseModel, ConfigDict
from pydantic_units import OpenMMQuantity

BoxShape = Literal["cube", "dodecahedron", "octahedron"]


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
    # Production MD length (ns) for the MMGBSA protomer-funnel equilibration used
    # by ``run_modbind`` to pick the protomer and the lowest-energy escape start.
    mmgbsa_equilibration_ns: float = 0.1

    # --- Sampling ---
    # High temperature for the BOUND (unbinding) simulation. Its populations are
    # reweighted back to ``reference_temperature`` with exponent T_b/T_ref (Eq. 4).
    bound_temperature: OpenMMQuantity[unit.kelvin] = 900.0 * unit.kelvin
    # Temperature for the UNBOUND (ligand-in-solvent) simulation. The paper (SI,
    # "Two-state ModBinddG Calculations") runs the ligand-only state at 300 K,
    # i.e. at the reference temperature, so its populations need NO reweighting
    # (exponent T_u/T_ref = 1). The bound and unbound states therefore carry
    # *different* reweighting exponents; ``reweight.py`` documents how the
    # temperature and escape-count differences between the states are handled in
    # the Eq. 14 ratio. Set this equal to ``temperature`` to recover the old
    # both-states-hot behaviour.
    unbound_temperature: OpenMMQuantity[unit.kelvin] = 300.0 * unit.kelvin
    # Target temperature that populations are reweighted back to (Eq. 4).
    reference_temperature: OpenMMQuantity[unit.kelvin] = 300.0 * unit.kelvin
    integrator_step_size: OpenMMQuantity[unit.picosecond] = 0.002 * unit.picoseconds
    friction: OpenMMQuantity[unit.picosecond**-1] = 1.0 / unit.picosecond  # type: ignore

    n_replicas: int = 8
    # 0.01 ns capture interval for BOTH states, matching the paper (SI-4: "record
    # frames every 0.01 ns... we apply the same parameters... to our unbound
    # term"). A single common interval means no cross-state frame-rate conversion
    # is needed (the factor in ``reweight.py`` is 1). NB: because the two states
    # carry different reweighting exponents, the Eq. 14 ratio still depends on
    # this interval as dt ** (bound_exp - unbound_exp) -- an intrinsic property of
    # the mixed-temperature model (the Eq. 3 constant C does not cancel across
    # states of different exponent) -- so it is fixed at the paper's 0.01 ns
    # rather than treated as a free knob.
    bound_frame_interval: OpenMMQuantity[unit.picosecond] = 1 * unit.picoseconds
    unbound_frame_interval: OpenMMQuantity[unit.picosecond] = 1 * unit.picoseconds
    # Upper bound on the length of any single escape trajectory.
    max_escape_time: OpenMMQuantity[unit.nanosecond] = 5.0 * unit.nanosecond
    # Target number of escape segments for the single long unbound simulation.
    unbound_target_escapes: int = 32
    # 10 ns matches the paper's ligand-in-solvent runs and leaves headroom to
    # reach ``unbound_target_escapes`` at 300 K, where diffusion (and hence the
    # escape rate) is slower than in the hot bound simulation.
    unbound_max_time: OpenMMQuantity[unit.nanosecond] = 10.0 * unit.nanosecond

    # --- State definitions / reweighting (Angstrom) ---
    bound_state_radius: float = 2.0
    absorbing_boundary_radius: float = 5.0
    unbound_radius: float = 5.0
    bin_size: float = 4.0
    standard_volume: float = 1661.0  # Angstrom^3 (1 M standard state)
    pmf_max_radius: float = 30.0

    # --- Statistics / output ---
    n_bootstrap: int = 1000
    random_seed: int = 1234

    @property
    def reweight_exponent(self) -> float:
        """Bound-state ``T_b / T_ref`` reweighting exponent (Eq. 4)."""
        t_sim = self.bound_temperature.value_in_unit(unit.kelvin)
        t_ref = self.reference_temperature.value_in_unit(unit.kelvin)
        return t_sim / t_ref

    @property
    def unbound_reweight_exponent(self) -> float:
        """Unbound-state ``T_u / T_ref`` reweighting exponent (Eq. 4).

        Equals 1 when the unbound state is run at the reference temperature (the
        default 300 K), i.e. its populations are already canonical and receive no
        reweighting. It differs from :attr:`reweight_exponent` whenever the two
        states are run at different temperatures.
        """
        t_unbound = self.unbound_temperature.value_in_unit(unit.kelvin)
        t_ref = self.reference_temperature.value_in_unit(unit.kelvin)
        return t_unbound / t_ref

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
            "temperature": str(self.bound_temperature),
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
        }
        hash_str = json.dumps(conf_dict, sort_keys=True)
        return xxhash.xxh64(hash_str.encode()).hexdigest()
