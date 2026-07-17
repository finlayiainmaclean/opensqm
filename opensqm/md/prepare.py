"""Module containing vanilla MD protocols."""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Sequence

from loguru import logger
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
from pydantic_units import OpenMMQuantity
from rdkit import Chem

from opensqm.md.rest import apply_rest
from opensqm.utils import LIGAND_FORCEFIELD_DIR

if TYPE_CHECKING:
    from pathlib import Path

    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper

logging.getLogger("openff.interchange.smirnoff").setLevel(logging.WARNING)

_PROTEIN_FORCEFIELD_FILES = (
    "amber/ff14SB.xml",
    "amber/phosaa10.xml",
    "amber/tip3p_HFE_multivalent.xml",
    "amber/tip3p_standard.xml",
)

_IMPLICIT_FORCEFIELD_FILES = (
    "amber/ff14SB.xml",
    "amber/phosaa10.xml",
    "implicit/gbn2.xml",
)

SolventMode = Literal["explicit", "implicit"]

PartialChargeMethod = Literal["nagl", "sqm"]
DEFAULT_PARTIAL_CHARGE_METHOD: PartialChargeMethod = "nagl"

# OpenFF's production graph-net AM1-BCC surrogate model.
NAGL_MODEL = "openff-gnn-am1bcc-1.0.0.pt"

_SOLVENT_RESIDUE_NAMES = frozenset(
    {"HOH", "WAT", "SOL", "TIP3", "TIP", "NA", "CL", "K", "MG", "ZN", "CA", "CS"}
)
_ION_SYMBOLS = frozenset({"Na", "Cl", "K", "Mg", "Zn", "Ca", "Cs"})


def strip_solvent(modeller: Modeller) -> Modeller:
    """Remove crystallographic waters and monatomic ions from a modeller."""
    to_delete = []
    for atom in modeller.topology.atoms():
        residue = atom.residue
        if residue.name in _SOLVENT_RESIDUE_NAMES:
            to_delete.append(atom)
            continue
        if len(list(residue.atoms())) == 1 and atom.element is not None:
            if atom.element.symbol in _ION_SYMBOLS:
                to_delete.append(atom)
    if not to_delete:
        return modeller
    cleaned = Modeller(modeller.topology, modeller.positions)
    cleaned.delete(to_delete)
    return cleaned


@lru_cache(maxsize=1)
def _nagl_charger() -> "tuple[NAGLToolkitWrapper, Path]":
    """Build the NAGL toolkit wrapper and resolve its AM1-BCC model path once.

    NAGL is OpenFF's cross-platform graph-net surrogate for AM1-BCC. Its native
    PyTorch backend needs no ``dgl``, so it works on macOS. Cached so the
    wrapper build and model lookup happen at most once per process.
    """
    from openff.nagl_models import validate_nagl_model_path
    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper

    return NAGLToolkitWrapper(), validate_nagl_model_path(NAGL_MODEL)


def _assign_sqm_charges(ligand: Molecule) -> None:
    """Assign AM1-BCC charges with AmberTools (the ``sqm`` semi-empirical engine)."""
    # Reuse the molecule's existing conformer(s) instead of letting the
    # toolkit regenerate one. OpenFF otherwise embeds a fresh ETKDG
    # conformer, and for large multi-titratable ligands (e.g. the CpH
    # protomers here, with aromatic-NH3+ / free -COOH groups) that can
    # seed a geometry whose AM1 minimisation never converges in sqm,
    # making antechamber abort with a fatal error. The pre-built
    # protomer conformers are well-formed, so passing them keeps sqm
    # well-behaved.
    use_conformers = ligand.conformers if ligand.n_conformers else None
    ligand.assign_partial_charges(
        "am1bcc",
        toolkit_registry=AmberToolsToolkitWrapper(),
        use_conformers=use_conformers,
    )


def assign_ligand_charges(
    ligand: Molecule,
    partial_charge_method: PartialChargeMethod = DEFAULT_PARTIAL_CHARGE_METHOD,
) -> None:
    """Assign partial charges to the ligand.

    ``nagl`` (the default) uses OpenFF's cross-platform GNN AM1-BCC model. It
    ships a chemical-domain check and raises ``ValueError`` for chemistry
    outside its training set (unusual bonds/elements); any such failure — or a
    backend error — falls back to ``sqm``. ``sqm`` runs AmberTools AM1-BCC
    directly, reusing the molecule's existing conformer(s) when present.
    """
    if ligand.partial_charges is not None:
        return
    match partial_charge_method:
        case "nagl":
            try:
                wrapper, model_path = _nagl_charger()
                ligand.assign_partial_charges(model_path, toolkit_registry=wrapper)
            except Exception as exc:
                # Fall back to sqm on any NAGL failure (not installed, backend
                # error, or chemistry outside the model's chemical domain).
                logger.warning(
                    f"NAGL charges failed ({type(exc).__name__}: {exc}); falling back to sqm"
                )
                _assign_sqm_charges(ligand)
        case "sqm":
            _assign_sqm_charges(ligand)


def get_ligand_forcefield(
    ligand: Molecule | list[Molecule],
    forcefield: ForceField | None = None,
    partial_charge_method: PartialChargeMethod = DEFAULT_PARTIAL_CHARGE_METHOD,
) -> ForceField:
    """Register a SMIRNOFF template generator per ligand on a ForceField.

    `ligand` may be a single OpenFF Molecule or a list of them (e.g. multiple
    protonation states of the same ligand for constant-pH simulations).
    Partial charges are assigned to each molecule if not already present.

    Each molecule is parameterised with the standard `openff-2.2.0.offxml`
    (Sage) forcefield and gets its own template generator registered on the
    forcefield; OpenMM's residue matching iterates through the registered
    generators and each one only claims its own molecule (via OpenFF's
    isomorphism check), so they coexist without conflict.

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
        smirnoff = SMIRNOFFTemplateGenerator(
            forcefield="openff-2.2.0.offxml",
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
    box_shape: str = "cube",
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


def _small_molecule_topology(
    molecule: Chem.Mol | Molecule,
    residue_name: str,
) -> tuple[Topology, unit.Quantity, Molecule]:
    if isinstance(molecule, Chem.Mol):
        offmol = Molecule.from_rdkit(molecule, allow_undefined_stereo=True)
    else:
        offmol = molecule

    lig_top = offmol.to_topology().to_openmm()
    lig_pos = (offmol.conformers[0].m * unit.angstrom).in_units_of(unit.nanometer)

    for chain in lig_top.chains():
        if not str(chain.id).strip():
            chain.id = "L"
        for res in chain.residues():
            res.name = residue_name

    return lig_top, lig_pos, offmol


def prepare_system(
    *,
    protein_modeller: Modeller | None = None,
    small_molecules: Sequence[tuple[Chem.Mol | Molecule, str]] | None = None,
    padding: float = 1.2,
    ionic_strength: float = 0.15,
    box_shape: str = "cube",
    solvent_mode: SolventMode = "explicit",
) -> tuple[Topology, unit.Quantity, ForceField]:
    """Build a system from optional small molecules and/or protein."""
    if not small_molecules:
        if protein_modeller is None:
            raise ValueError(
                "prepare_system requires protein_modeller when small_molecules is empty"
            )
        if solvent_mode == "implicit":
            raise ValueError("implicit solvent requires at least one small molecule")
        return prepare_protein(
            protein_modeller,
            padding=padding,
            ionic_strength=ionic_strength,
        )

    offmols: list[Molecule] = []
    modeller: Modeller | None = None
    for molecule, residue_name in small_molecules:
        lig_top, lig_pos, offmol = _small_molecule_topology(molecule, residue_name)
        offmols.append(offmol)
        if modeller is None:
            modeller = Modeller(lig_top, lig_pos)
        else:
            modeller.add(lig_top, lig_pos)

    forcefield = get_ligand_forcefield(offmols)
    if solvent_mode == "explicit":
        forcefield.loadFile(_PROTEIN_FORCEFIELD_FILES)
    else:
        forcefield.loadFile(_IMPLICIT_FORCEFIELD_FILES)

    if protein_modeller is not None:
        if solvent_mode == "implicit":
            protein_modeller = strip_solvent(
                Modeller(protein_modeller.topology, protein_modeller.positions)
            )
        modeller.add(protein_modeller.topology, protein_modeller.positions)

    if solvent_mode == "explicit":
        modeller.addSolvent(
            forcefield,
            ionicStrength=ionic_strength * unit.molar,
            padding=padding * unit.nanometers,
            boxShape=box_shape,
            positiveIon="Na+",
            negativeIon="Cl-",
        )
    return modeller.topology, modeller.positions, forcefield


def prepare_complex(
    ligand: Chem.Mol | Molecule,
    padding: float = 1.2,
    protein_modeller: Modeller | None = None,
    box_shape: str = "cube",
    solvent_mode: SolventMode = "explicit",
) -> tuple[Topology, unit.Quantity, ForceField]:
    """Prepare the complex by building ligand and protein into a modeller."""
    return prepare_system(
        protein_modeller=protein_modeller,
        small_molecules=[(ligand, "LIG")],
        padding=padding,
        box_shape=box_shape,
        solvent_mode=solvent_mode,
    )


def build_complex_forcefield(
    ligand: Chem.Mol | Molecule,
    *,
    solvent_mode: SolventMode = "explicit",
) -> ForceField:
    """Build a ForceField for an already-assembled ligand-protein topology.

    Registers a SMIRNOFF template generator for the ligand chemistry and loads
    the protein/water (or implicit) parameter sets, but does **no** solvation or
    assembly. Use this to call :func:`create_system` on a pre-built topology
    (e.g. a pre-equilibrated snapshot) instead of rebuilding it from scratch.
    """
    if isinstance(ligand, Molecule):
        offmol = ligand
    else:
        offmol = Molecule.from_rdkit(ligand, allow_undefined_stereo=True)
    forcefield = get_ligand_forcefield([offmol])
    if solvent_mode == "explicit":
        forcefield.loadFile(_PROTEIN_FORCEFIELD_FILES)
    else:
        forcefield.loadFile(_IMPLICIT_FORCEFIELD_FILES)
    return forcefield


def prepare_protein(
    protein_modeller: Modeller,
    padding: float = 1.2,
    ionic_strength: float = 0.15,
) -> tuple[Topology, unit.Quantity, ForceField]:
    """Solvate a protein in a water/ion box (no ligand)."""
    forcefield = app.ForceField(*_PROTEIN_FORCEFIELD_FILES)
    modeller = Modeller(protein_modeller.topology, protein_modeller.positions)
    modeller.addSolvent(
        forcefield,
        ionicStrength=ionic_strength * unit.molar,
        padding=padding * unit.nanometers,
        boxShape="cube",
        positiveIon="Na+",
        negativeIon="Cl-",
    )
    return modeller.topology, modeller.positions, forcefield


def create_system(
    forcefield: ForceField,
    topology: Topology,
    rest_ligand: bool = False,
    *,
    implicit_solvent: bool = False,
) -> System:
    """Create an OpenMM System from the forcefield and topology."""
    if implicit_solvent:
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=2.0 * unit.nanometers,
            constraints=app.HBonds,
        )
    else:
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=9.0 * unit.angstroms,
            # switchDistance=8.0 * unit.angstroms,
            constraints=app.HBonds,
            rigidWater=True,
            hydrogenMass=2.0 * unit.dalton,
        )

    if rest_ligand:
        ligand_idxs = {atom.index for atom in topology.atoms() if atom.residue.name == "LIG"}
        if ligand_idxs:
            apply_rest(system, ligand_idxs)

    system.addForce(CMMotionRemover())
    return system


def create_integrator(
    integrator_ps_per_step: OpenMMQuantity[unit.picosecond],
    temperature: OpenMMQuantity[unit.kelvin] = 300 * unit.kelvin,
) -> LangevinMiddleIntegrator:
    """Create a Langevin Middle Integrator at the given temperature."""
    return LangevinMiddleIntegrator(temperature, 1 / unit.picosecond, integrator_ps_per_step)
