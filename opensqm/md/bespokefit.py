"""Module containing bespoke forcefield generation logic."""

import multiprocessing
import os

from loguru import logger
from openff.bespokefit.executor import BespokeExecutor, BespokeWorkerConfig
from openff.bespokefit.executor.client import BespokeFitClient, Settings
from openff.bespokefit.workflows import BespokeWorkflowFactory
from openff.qcsubmit.common_structures import QCSpec
from openff.toolkit.topology import Molecule

from opensqm.utils import LIGAND_FORCEFIELD_DIR

NUM_CPUS = os.environ.get("NUM_CPUS", multiprocessing.cpu_count())


def generate_bespoke_offxml(ligand: Molecule, overwrite: bool = False) -> str | None:
    """
    Generate a bespoke OpenFF forcefield (.offxml) for a given RDKit molecule.

    Saves the file as <inchikey>.offxml and returns the generated ForceField object.

    Args:
        rdmol: The RDKit molecule to parameterize.
        output_dir: Directory to save the resulting .offxml file.

    Returns
    -------
        The openff.toolkit.typing.engines.smirnoff.ForceField object if successful,
        otherwise None.
    """
    # Generate the InChIKey for the filename from the RDKit Mol
    inchikey = ligand.to_inchikey(fixed_hydrogens=True)

    output_filename = LIGAND_FORCEFIELD_DIR / f"{inchikey}.offxml"
    failed_output_filename = LIGAND_FORCEFIELD_DIR / f"{inchikey}.failed"

    if output_filename.exists() and not overwrite:
        logger.info(f"Bespoke parameters for {inchikey} already exist: {output_filename}")
        return output_filename

    if failed_output_filename.exists():
        logger.info(f"Bespoke parameters for {inchikey} failed: {failed_output_filename}")
        return None

    # Create the factory with default workflows (it will use xtb if it is set as default)
    factory = BespokeWorkflowFactory(
        default_qc_specs=[
            QCSpec(
                method="gfn2xtb",
                basis=None,
                program="xtb",
                spec_name="xtb",
                spec_description="gfn2xtb",
            )
        ]
    )
    workflow_schema = factory.optimization_schema_from_molecule(molecule=ligand)

    # Create a client to interface with the executor
    settings = Settings()
    client = BespokeFitClient(settings=settings)

    logger.info(f"Submitting bespoke fit for {inchikey}...")
    with BespokeExecutor(
        n_fragmenter_workers=1,
        n_optimizer_workers=1,
        n_qc_compute_workers=1,
        qc_compute_worker_config=BespokeWorkerConfig(n_cores=NUM_CPUS),
    ):
        # Submit our workflow to the executor
        task_id = client.submit_optimization(input_schema=workflow_schema)
        # Wait until the executor is done
        output = client.wait_until_complete(task_id)

    if output.status == "success":
        # Save the resulting force field to the OFFXML file
        output.bespoke_force_field.to_file(output_filename)
        logger.info(f"Successfully generated bespoke parameters: {output_filename}")
        return output_filename
    elif output.status == "errored":
        logger.error(f"BespokeFit failed for {inchikey}: {output.error}")
        failed_output_filename.touch()
        return None
