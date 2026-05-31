"""Module containing vanilla MD protocols."""

import logging
from typing import Literal

from openff.toolkit.topology import Molecule  # type: ignore
from openff.toolkit.utils.toolkits import AmberToolsToolkitWrapper  # type: ignore
from openmm import (
    CMMotionRemover,
    LangevinMiddleIntegrator,
    System,
    app,
    unit,
)
from openmm.app import Modeller
from openmm.app.forcefield import ForceField
from openmm.app.topology import Topology
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from rdkit import Chem
from loguru import logger
from opensqm.md.bespokefit import generate_bespoke_offxml
from opensqm.md.rest import apply_rest
from opensqm.utils import LIGAND_FORCEFIELD_DIR
from pathlib import Path
from pydantic_units import OpenMMQuantity

logging.getLogger("openff.interchange.smirnoff").setLevel(logging.WARNING)


def assign_ligand_charges(
    ligand: Molecule,
    partial_charge_method: Literal["am1bcc"] = "am1bcc",
) -> None:
    """Assign partial charges to the ligand."""
    if ligand.partial_charges is None:
        match partial_charge_method:
            case "am1bcc":
                toolkit_registry = AmberToolsToolkitWrapper()
                ligand.assign_partial_charges("am1bcc", toolkit_registry=toolkit_registry)


def get_ligand_forcefield(
    ligand: Molecule | list[Molecule],
    bespoke_ligand_forcefield: bool = False,
    forcefield: ForceField | None = None,
    partial_charge_method: Literal["am1bcc"] = "am1bcc",
) -> ForceField:
    """Register a SMIRNOFF template generator per ligand on a ForceField.

    `ligand` may be a single OpenFF Molecule or a list of them (e.g. multiple
    protonation states of the same ligand for constant-pH simulations).
    Partial charges are assigned to each molecule if not already present.

    When `bespoke_ligand_forcefield` is True, `generate_bespoke_offxml` is
    called once per molecule and the resulting OFFXML is used as the base
    forcefield for that molecule's `SMIRNOFFTemplateGenerator`. If bespoke
    fitting fails or returns nothing, the molecule falls back to the standard
    `openff-2.2.0.offxml`. Each molecule gets its own template generator
    registered on the forcefield; OpenMM's residue matching iterates through
    the registered generators and each one only claims its own molecule (via
    OpenFF's isomorphism check), so they coexist without conflict.

    If `forcefield` is None, a fresh empty `app.ForceField()` is created and
    returned. Otherwise the SMIRNOFF generators are registered on the supplied
    forcefield (mutating it) and the same forcefield is returned for chaining.
    """
    ligands = [ligand] if isinstance(ligand, Molecule) else list(ligand)
    for lig in ligands:
        assign_ligand_charges(lig, partial_charge_method)

    if forcefield is None:
        forcefield = app.ForceField()

    standard_smirnoff_cache = (LIGAND_FORCEFIELD_DIR / "smirnoff.json").resolve()

    for lig in ligands:
        ligand_forcefield_file: str | None = None
        if bespoke_ligand_forcefield:
            ligand_forcefield_file = None
            
            # try:
            #     bespoke_path = generate_bespoke_offxml(lig)
            # except Exception as e:
            #     logger.error(f"Failed to generate bespoke forcefield for {lig.to_smiles()}: {e}")
            #     bespoke_path = None
            # ligand_forcefield_file = (
            #     str(Path(bespoke_path).resolve()) if bespoke_path else None
            # )

        smirnoff = SMIRNOFFTemplateGenerator(
            forcefield=ligand_forcefield_file or "openff-2.2.0.offxml",
            molecules=lig,
            cache=str(standard_smirnoff_cache),
        )
        forcefield.registerTemplateGenerator(smirnoff.generator)

    return forcefield




def solvate_ligand(
    ligand: Chem.Mol,
    forcefield: ForceField,
    ionic_strength: float = 0.15,
    padding: float = 1.5,
    box_shape: str = "dodecahedron",
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
    residue_name: str = "LIG",
) -> tuple[Topology, unit.Quantity]:
    """Solvate a single ligand in a water/ion box using a pre-built forcefield.

    The forcefield must already have a template generator registered for the
    ligand chemistry (e.g. via :func:`get_ligand_forcefield` or by registering
    a :class:`SMIRNOFFTemplateGenerator` on it directly). The returned
    topology has the ligand as residue 0, followed by the solvent residues
    added by `Modeller.addSolvent`, and carries periodic box vectors.

    The default 1.5 nm padding is a touch more generous than the protein
    default in :func:`prepare_complex` so that even very small ligands
    produce a periodic box wide enough for typical 0.9-1.0 nm PME cutoffs
    in a rhombic dodecahedron geometry.
    """
    offmol = Molecule.from_rdkit(ligand, allow_undefined_stereo=True)

    lig_top = offmol.to_topology().to_openmm()
    lig_pos = (offmol.conformers[0].m * unit.angstrom).in_units_of(unit.nanometer)

    for chain in lig_top.chains():
        for res in chain.residues():
            res.name = residue_name

    modeller = Modeller(lig_top, lig_pos)
    modeller.addSolvent(
        forcefield,
        ionicStrength=ionic_strength * unit.molar,
        padding=padding * unit.nanometers,
        boxShape=box_shape,
        positiveIon=positive_ion,
        negativeIon=negative_ion,
    )
    return modeller.topology, modeller.positions


def prepare_complex(
    ligand: Chem.Mol | Molecule, bespoke_ligand_forcefield: bool = True,
    padding: float = 1.2,
    protein_modeller: Modeller | None = None,
) -> tuple[Topology, unit.Quantity, ForceField]:
    """Prepare the complex by building ligand and protein into a modeller."""

    if isinstance(ligand, Chem.Mol):
        offmol = Molecule.from_rdkit(ligand, allow_undefined_stereo=True)
    else:
        offmol = ligand

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
    if protein_modeller is not None:
        modeller.add(protein_modeller.topology, protein_modeller.positions)

    modeller.addSolvent(
        forcefield,
        ionicStrength=0.15 * unit.molar,
        padding=padding * unit.nanometers,
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



def create_integrator(integrator_ps_per_step: OpenMMQuantity[unit.picosecond]) -> LangevinMiddleIntegrator:
    """Create a Langevin Middle Integrator."""
    return LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        integrator_ps_per_step
    )
