from typing import Any
from pydantic import BaseModel, Field, validator

from openmm.app import ForceField, PME, CutoffNonPeriodic, HBonds
from openmm import unit
from openmm import LangevinIntegrator
import xxhash
import json


def _default_explicit_params() -> dict:
    return dict(
        nonbondedMethod=PME,
        nonbondedCutoff=0.9 * unit.nanometers,
        constraints=HBonds,
        hydrogenMass=1.5 * unit.dalton,
    )


def _default_implicit_params() -> dict:
    return dict(
        nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=2.0 * unit.nanometers,
        constraints=HBonds,
    )


class SimulationConfig(BaseModel):

    explicit_ff_files: tuple = ('amber14-all.xml', 'amber14/tip3pfb.xml')
    implicit_ff_files: tuple = ('amber14-all.xml', 'implicit/gbn2.xml')
    explicit_params: dict = Field(default_factory=_default_explicit_params)
    implicit_params: dict = Field(default_factory=_default_implicit_params)
    temperature: float = 300.0  # Kelvin
    relaxation_steps: int = 100
    timestep: float = 0.004     # ps
    relaxation_timestep: float = 0.002  # ps
    friction: float = 1.0       # ps^-1
    relaxation_friction: float = 10.0   # ps^-1


    # Derived attributes
    explicit_ff: Any = None
    implicit_ff: Any = None
    integrator: Any = None
    relaxation_integrator: Any = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._make_forcefields()
        self._make_integrators()

    def _make_forcefields(self):
        """Instantiate OpenMM ForceField objects."""
        object.__setattr__(self, 'explicit_ff', ForceField(*self.explicit_ff_files))
        object.__setattr__(self, 'implicit_ff', ForceField(*self.implicit_ff_files))

    def _make_integrators(self):
        """Create production and relaxation integrators."""
        temperature = self.temperature * unit.kelvin
        object.__setattr__(self, 'integrator', LangevinIntegrator(
            temperature,
            self.friction / unit.picosecond,
            self.timestep * unit.picoseconds))
        object.__setattr__(self, 'relaxation_integrator', LangevinIntegrator(
            temperature,
            self.relaxation_friction / unit.picosecond,
            self.relaxation_timestep * unit.picoseconds))

    def hash(self) -> str:
        """Create a reproducible hash of parameters that affect reference energies.

        Only force field definitions and temperature are included, since
        sampling dynamics parameters (pH, timestep, friction) do not change
        the converged reference energy values.
        """
        conf_dict = {
            'explicit_ff_files': list(self.explicit_ff_files),
            'implicit_ff_files': list(self.implicit_ff_files),
            'explicit_params': {k: str(v) for k, v in self.explicit_params.items()},
            'implicit_params': {k: str(v) for k, v in self.implicit_params.items()},
            'temperature': self.temperature,
        }
        hash_str = json.dumps(conf_dict, sort_keys=True)
        return xxhash.xxh64(hash_str.encode()).hexdigest()


# Example use
if __name__=="__main__":
    config = SimulationConfig()
    print("Hash:", config.hash())
    print("Explicit FF:", config.explicit_ff)
    print("Integrator:", config.integrator)