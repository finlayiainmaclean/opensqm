"""Module containing vanilla MD protocols."""

import logging
from typing import Literal

# pyrefly: ignore [missing-import]
from espaloma_charge.openff_wrapper import (
    EspalomaChargeToolkitWrapper,  # type: ignore
)
# from mdtop import Topology as MDTopology
# pyrefly: ignore [missing-import]
from openff.toolkit.topology import Molecule  # type: ignore
# pyrefly: ignore [missing-import]
from openff.toolkit.utils.toolkits import AmberToolsToolkitWrapper  # type: ignore
# pyrefly: ignore [missing-import]
from openmm import (
    CMMotionRemover,
    LangevinMiddleIntegrator,
    System,
    app,
    unit,
)
# pyrefly: ignore [missing-import]
from openmm.app import Modeller
# pyrefly: ignore [missing-import]
from openmm.app.forcefield import ForceField
# pyrefly: ignore [missing-import]
from openmm.app.topology import Topology
# pyrefly: ignore [missing-import]
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
# pyrefly: ignore [missing-import]
from rdkit import Chem

from opensqm.md.bespokefit import generate_bespoke_offxml
from opensqm.md.rest import apply_rest
from opensqm.utils import LIGAND_FORCEFIELD_DIR
from pathlib import Path

logging.getLogger("openff.interchange.smirnoff").setLevel(logging.WARNING)


def assign_ligand_charges(
    ligand: Molecule,
    partial_charge_method: Literal["am1bcc", "espaloma-am1bcc"] = "espaloma-am1bcc",
) -> None:
    """Assign partial charges to the ligand."""
    if ligand.partial_charges is None:
        match partial_charge_method:
            case "am1bcc":
                toolkit_registry = AmberToolsToolkitWrapper()
                ligand.assign_partial_charges("am1bcc", toolkit_registry=toolkit_registry)
            case "espaloma-am1bcc":
                toolkit_registry = EspalomaChargeToolkitWrapper()
                ligand.assign_partial_charges("espaloma-am1bcc", toolkit_registry=toolkit_registry)


def get_ligand_forcefield(
    ligand: Molecule,
    bespoke_ligand_forcefield: bool = True,
    partial_charge_method: Literal["am1bcc", "espaloma-am1bcc"] = "espaloma-am1bcc",
) -> ForceField | None:
    """Get the ligand forcefield."""
    assign_ligand_charges(ligand, partial_charge_method)

    standard_smirnoff_cache = (LIGAND_FORCEFIELD_DIR / "smirnoff.json").resolve()

    ligand_forcefield_file = None
    if bespoke_ligand_forcefield:
        ligand_forcefield_file = generate_bespoke_offxml(ligand)
        ligand_forcefield_file = (
            str(Path(ligand_forcefield_file).resolve()) if ligand_forcefield_file else None
        )

    forcefield = app.ForceField()
    smirnoff = SMIRNOFFTemplateGenerator(
        forcefield=ligand_forcefield_file or "openff-2.2.0.offxml",
        molecules=ligand,
        cache=str(standard_smirnoff_cache),
    )
    forcefield.registerTemplateGenerator(smirnoff.generator)

    return forcefield


def prepare_complex(
    ligand: Chem.Mol, protein_modeller: Modeller, bespoke_ligand_forcefield: bool = True
) -> tuple[Topology, unit.Quantity, ForceField]:
    """Prepare the complex by building ligand and protein into a modeller."""
    offmol = Molecule.from_rdkit(ligand, allow_undefined_stereo=False)

    forcefield = get_ligand_forcefield(offmol, bespoke_ligand_forcefield)

    if forcefield is None:
        raise ValueError(f"Failed to create ligand forcefield for {Chem.MolToSmiles(ligand)}")

    files = (
        "amber/ff14SB.xml",
        "amber/phosaa10.xml",
        "amber/tip3p_HFE_multivalent.xml",
        "amber/tip3p_standard.xml",
    )
    forcefield.loadFile(files)

    lig_top = offmol.to_topology().to_openmm()
    lig_pos = (offmol.conformers[0].m * unit.angstrom).in_units_of(unit.nanometer)

    for chain in lig_top.chains():
        for res in chain.residues():
            res.name = "LIG"

    modeller = Modeller(lig_top, lig_pos)
    modeller.add(protein_modeller.topology, protein_modeller.positions)

    modeller.addSolvent(
        forcefield,
        ionicStrength=0.15 * unit.molar,
        padding=1.0 * unit.nanometers,
        boxShape="dodecahedron",
        positiveIon="Na+",
        negativeIon="Cl-",
    )

    return modeller.topology, modeller.positions, forcefield


def create_system(forcefield: ForceField, topology: Topology, rest_ligand: bool = False) -> System:
    """Create an OpenMM System from the forcefield and topology."""
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=10.0 * unit.angstroms,
        switchDistance=9.0 * unit.angstroms,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=1.5,
    )

    if rest_ligand:
        ligand_idxs = {atom.index for atom in topology.atoms() if atom.residue.name == "LIG"}
        if ligand_idxs:
            apply_rest(system, ligand_idxs)

    system.addForce(CMMotionRemover())
    return system


def create_implicit_system(
    forcefield: ForceField, topology: Topology, positions: unit.Quantity, rest_ligand: bool = False
) -> tuple[System, Topology, unit.Quantity]:
    """Create an OpenMM System in implicit solvent by removing explicit solvent and ions."""
    forcefield.loadFile("implicit/obc2.xml")

    modeller = Modeller(topology, positions)
    to_delete = [
        r
        for r in modeller.topology.residues()
        if r.name in ("HOH", "WAT", "SOL", "NA", "CL", "K", "MG")
    ]
    modeller.delete(to_delete)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=1.5,
    )

    if rest_ligand:
        ligand_idxs = {
            atom.index for atom in modeller.topology.atoms() if atom.residue.name == "LIG"
        }
        if ligand_idxs:
            apply_rest(system, ligand_idxs)

    system.addForce(CMMotionRemover())
    return system, modeller.topology, modeller.positions


def create_integrator(integrator_ps_per_step: float) -> LangevinMiddleIntegrator:
    """Create a Langevin Middle Integrator."""
    return LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        integrator_ps_per_step * unit.picoseconds,
    )
