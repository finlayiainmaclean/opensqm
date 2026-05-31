"""Run reference-energy generation from the command line.

Equivalent to the old ``python opensqm/cph/reference_energy.py`` entry
point. Invoke as::

    python -m opensqm.cph.reference_energy

Generates (or loads from cache) a :class:`TitratableResidueReference`
for every entry in :data:`opensqm.cph.reference_energy.model_compounds.MODEL_COMPOUNDS`
under the default :class:`opensqm.cph.simulation_config.SimulationConfig`.
"""
# pyrefly: ignore [missing-import]
from loguru import logger

from opensqm.cph.simulation_config import ConstantpHSettings

from .generate import generate_residue_reference_dict


def main() -> None:
    config = ConstantpHSettings()
    logger.info(f"Hash: {config.hash()}")

    references = generate_residue_reference_dict(config)
    for residue_name, reference in references.items():
        logger.info(f"{residue_name}: {reference}")


if __name__ == "__main__":
    main()
