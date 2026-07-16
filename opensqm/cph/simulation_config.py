"""Pydantic configuration model for constant-pH force fields and integrators."""

import json
from typing import TYPE_CHECKING, Any, Literal

import xxhash
from openmm import LangevinIntegrator, unit
from openmm.app import PME, CutoffNonPeriodic, ForceField, HBonds
from pydantic import BaseModel, Field
from pydantic_units import OpenMMQuantity

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


def _default_explicit_params() -> dict:
    return {
        "nonbondedMethod": PME,
        "nonbondedCutoff": 0.9 * unit.nanometers,
        "constraints": HBonds,
        "hydrogenMass": 1.5 * unit.dalton,
    }


def _default_implicit_params() -> dict:
    return {
        "nonbondedMethod": CutoffNonPeriodic,
        "nonbondedCutoff": 2.0 * unit.nanometers,
        "constraints": HBonds,
    }


class ConstantpHSettings(BaseModel):
    """Force-field, integrator, and system settings shared across constant-pH replicas."""

    explicit_ff_files: tuple = ("amber14-all.xml", "amber14/tip3pfb.xml")
    implicit_ff_files: tuple = ("amber14-all.xml", "implicit/gbn2.xml")

    # Ligand parameterisation (used when ligands are passed to generate_references).
    # "nagl" (the default) is OpenFF's cross-platform GNN AM1-BCC model; it falls
    # back to "sqm" (AmberTools AM1-BCC) for chemistry outside its training domain.
    # See opensqm.md.prepare.assign_ligand_charges.
    partial_charge_method: Literal["nagl", "sqm"] = "nagl"

    explicit_params: dict = Field(default_factory=_default_explicit_params)
    implicit_params: dict = Field(default_factory=_default_implicit_params)
    temperature: OpenMMQuantity[unit.kelvin] = 300 * unit.kelvin
    relaxation_steps: int = 100
    timestep: OpenMMQuantity[unit.picosecond] = 0.004 * unit.picoseconds
    relaxation_timestep: OpenMMQuantity[unit.picosecond] = 0.002 * unit.picoseconds
    friction: OpenMMQuantity[unit.picosecond**-1] = 1.0 / unit.picosecond
    relaxation_friction: OpenMMQuantity[unit.picosecond**-1] = 10.0 / unit.picosecond

    # Derived attributes
    integrator: Any = None
    relaxation_integrator: Any = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._make_integrators()

    def make_explicit_ff(self) -> ForceField:
        """Build a fresh OpenMM ForceField for the explicit-solvent system.

        Forcefields are built on demand rather than cached on the config so
        that callers can register additional template generators (e.g. for
        ligands) on the returned object without mutating shared state.
        """
        return ForceField(*self.explicit_ff_files)

    def make_implicit_ff(self) -> ForceField:
        """Build a fresh OpenMM ForceField for the implicit-solvent system."""
        return ForceField(*self.implicit_ff_files)

    def get_explicit_forcefield(
        self, ligands: "Molecule | list[Molecule] | None" = None
    ) -> ForceField:
        """Build the explicit-solvent forcefield, optionally with ligand templates.

        Parameters
        ----------
        ligands : openff.toolkit.topology.Molecule or list of Molecule, optional
            Titratable ligand variant(s) whose SMIRNOFF templates should be
            registered on the returned forcefield via
            :func:`opensqm.md.prepare.get_ligand_forcefield`. When ``None``
            the result is identical to :meth:`make_explicit_ff`.
        """
        ff = self.make_explicit_ff()
        if ligands is None:
            return ff
        from opensqm.md.prepare import get_ligand_forcefield

        return get_ligand_forcefield(
            ligands,
            forcefield=ff,
            partial_charge_method=self.partial_charge_method,
        )

    def get_implicit_forcefield(
        self, ligands: "Molecule | list[Molecule] | None" = None
    ) -> ForceField:
        """Build the implicit-solvent forcefield, optionally with ligand templates.

        See :meth:`get_explicit_forcefield` for the ligand semantics.
        """
        ff = self.make_implicit_ff()
        if ligands is None:
            return ff
        from opensqm.md.prepare import get_ligand_forcefield

        return get_ligand_forcefield(
            ligands,
            forcefield=ff,
            partial_charge_method=self.partial_charge_method,
        )

    def _make_integrators(self) -> None:
        """Create production and relaxation integrators."""
        object.__setattr__(
            self, "integrator", LangevinIntegrator(self.temperature, self.friction, self.timestep)
        )
        object.__setattr__(
            self,
            "relaxation_integrator",
            LangevinIntegrator(
                self.temperature, self.relaxation_friction, self.relaxation_timestep
            ),
        )

    def hash(self) -> str:
        """Create a reproducible hash of parameters that affect reference energies.

        Includes force field definitions, temperature, and the ligand
        partial charge method since those change the converged reference
        energies. Sampling dynamics parameters (pH, timestep, friction)
        are intentionally excluded because they do not affect the
        converged values.
        """
        conf_dict = {
            "explicit_ff_files": list(self.explicit_ff_files),
            "implicit_ff_files": list(self.implicit_ff_files),
            "explicit_params": {k: str(v) for k, v in self.explicit_params.items()},
            "implicit_params": {k: str(v) for k, v in self.implicit_params.items()},
            "temperature": str(self.temperature),
            "partial_charge_method": self.partial_charge_method,
        }
        hash_str = json.dumps(conf_dict, sort_keys=True)
        return xxhash.xxh64(hash_str.encode()).hexdigest()


# Example use
if __name__ == "__main__":
    config = ConstantpHSettings()
    print("Hash:", config.hash())
    print("Explicit FF:", config.make_explicit_ff())
    print("Integrator:", config.integrator)
