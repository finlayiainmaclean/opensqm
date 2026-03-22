# ruff: noqa: D100, D103
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from opensqm.mopac.charges import get_mopac_formal_charges
from opensqm.mopac.constants import OptMode
from opensqm.mopac.geometry import annotate_mopac_formal_charges
from opensqm.mopac.ligand import get_correct_ligand
from opensqm.mopac.nitro import strip_mopac_nitro_aux_bonds
from opensqm.mopac.opt_mask import get_opt_mask
from opensqm.mopac.optimize import run_opt_from_rdmol
from opensqm.mopac.singlepoint import run_singlepoint_from_rdmol
from opensqm.rdkit_utils import get_coordinates, set_coordinates


def run_interaction_energy(*, ligand: Chem.Mol, protein: Chem.Mol) -> dict[str, float]:
    # Get ligand with annoted formal charges and pi bonds for mopac
    ligand, _dE = get_correct_ligand(ligand)

    rdkit_ligand_charge = Chem.GetFormalCharge(ligand)
    n_ligand_atoms = ligand.GetNumAtoms()

    # Get ligand charges from both rdkit and mopac and check
    # rdkit_ligand_formal_charges = get_rdkit_formal_charges(ligand)
    mopac_ligand_formal_charge, mopac_ligand_formal_charges = get_mopac_formal_charges(ligand)

    if mopac_ligand_formal_charge != rdkit_ligand_charge:
        print("RDKit and MOPAC disagree on ligand charge")
        Chem.MolToMolFile(ligand, "/tmp/ligand.mol")
        raise ValueError("RDKit and MOPAC disagree on ligand charge")

    # RDKit does not parse salt bridges well, so use MOPAC to find atom formal charges
    mopac_protein_charge, mopac_protein_formal_charges = get_mopac_formal_charges(protein)
    # Annotate the protein with these charges (strictly unneccesary as this is a round-trip
    annotate_mopac_formal_charges(protein, mopac_protein_formal_charges)
    complex_charge = rdkit_ligand_charge + mopac_protein_charge

    # Build the complex, making sure the ligand is first, as mopac will use the 1-indexed pi bonds
    complex = Chem.CombineMols(ligand, protein)

    # Annotate the complex with the rdkit-derived formal charges of the ligand
    # and the mopac-derived charges of the protein.
    # We need to shift the atom indices of the protein by N_atoms of the ligand
    # TODO(fin): Split cofactors to be treated like the ligand
    complex_formal_charges = mopac_ligand_formal_charges
    for protein_atom_ix, protein_atom_charge in mopac_protein_formal_charges.items():
        complex_atom_ix = protein_atom_ix + n_ligand_atoms
        complex_formal_charges[complex_atom_ix] = protein_atom_charge
    annotate_mopac_formal_charges(complex, complex_formal_charges)

    E_ligand = run_singlepoint_from_rdmol(
        ligand, use_mozyme=True, solvent="cosmo2", charge=rdkit_ligand_charge
    )
    E_protein = run_singlepoint_from_rdmol(
        protein, use_mozyme=True, solvent="cosmo2", charge=mopac_protein_charge
    )
    E_complex = run_singlepoint_from_rdmol(
        complex, use_mozyme=True, solvent="cosmo2", charge=complex_charge
    )
    dE_int = E_complex - E_protein - E_ligand

    scores = {
        "dE_int": dE_int,
        "E_complex": E_complex,
        "E_protein": E_protein,
        "E_ligand": E_ligand,
    }

    return scores


def optimise_complex(
    *,
    ligand: Chem.Mol,
    protein: Chem.Mol,
    mode: OptMode,
    gnorm: float = 1.0,
    use_rapid: bool = False,
    num_epochs: int = 30,
) -> tuple[Chem.Mol, Chem.Mol]:
    mopac_keywords = [
        "PM6-D3H4X",
        "MOZYME",
        f"LET({num_epochs})",
        "LBFGS",
        "RHF",
        "METAL",
        "NOMM",
        "EPS=78.5",
        f"GNORM={gnorm}",
    ]

    if use_rapid:
        mopac_keywords.append("RAPID")

    ligand_in = Chem.Mol(ligand)

    # Get ligand with annoted formal charges and pi bonds for mopac
    ligand, _dE = get_correct_ligand(ligand_in)
    ligand_pre_opt = Chem.Mol(ligand)

    rdkit_ligand_charge = Chem.GetFormalCharge(ligand)
    n_ligand_atoms = ligand.GetNumAtoms()

    # Get ligand charges from both rdkit and mopac and check
    mopac_ligand_formal_charge, mopac_ligand_formal_charges = get_mopac_formal_charges(ligand)
    assert mopac_ligand_formal_charge == rdkit_ligand_charge, (
        "RDKit and MOPAC disagree on ligand charge"
    )

    # RDKit does not parse salt bridges well, so use MOPAC to find atom formal charges
    mopac_protein_charge, mopac_protein_formal_charges = get_mopac_formal_charges(protein)
    # Annotate the protein with these charges (strictly unneccesary as this is a round-trip
    annotate_mopac_formal_charges(protein, mopac_protein_formal_charges)
    complex_charge = rdkit_ligand_charge + mopac_protein_charge

    # Build the complex, making sure the ligand is first, as mopac will use the 1-indexed pi bonds
    complex = Chem.CombineMols(ligand, protein)

    # AllChem.MMFFOptimizeMolecule(complex)

    # Annotate the complex with the rdkit-derived formal charges of the ligand
    # and the mopac-derived charges of the protein.
    # We need to shift the atom indices of the protein by N_atoms of the ligand
    # TODO(fin): Split cofactors to be treated like the ligand
    complex_formal_charges = mopac_ligand_formal_charges
    for protein_atom_ix, protein_atom_charge in mopac_protein_formal_charges.items():
        complex_atom_ix = protein_atom_ix + n_ligand_atoms
        complex_formal_charges[complex_atom_ix] = protein_atom_charge
    annotate_mopac_formal_charges(complex, complex_formal_charges)

    opt_mask = get_opt_mask(complex, mode=mode)
    logger.info(f"{sum(opt_mask)} atoms to optimise")

    complex_optimised = run_opt_from_rdmol(
        complex, opt_mask=opt_mask, mopac_keywords=mopac_keywords, charge=complex_charge
    )
    # Drop MOPAC-only O-O nitro links so RMSD / RDKit see the same graph as pre-MOPAC.
    complex_optimised = strip_mopac_nitro_aux_bonds(complex_optimised)

    # Extract ligand and protein conformations from the optimised complex
    ligand_idxs = [
        atom.GetIdx()
        for atom in complex_optimised.GetAtoms()
        if atom.GetPDBResidueInfo().GetResidueName() == "LIG"
    ]

    protein_idxs = [
        atom.GetIdx()
        for atom in complex_optimised.GetAtoms()
        if atom.GetPDBResidueInfo().GetResidueName() != "LIG"
    ]

    rw = Chem.RWMol(complex_optimised)
    # Remove all ligand atoms to get protein
    for idx in sorted(ligand_idxs, reverse=True):
        rw.RemoveAtom(idx)
    protein_opt = rw.GetMol()

    rw = Chem.RWMol(complex_optimised)
    # Remove all protein atoms to get ligand
    for idx in sorted(protein_idxs, reverse=True):
        rw.RemoveAtom(idx)
    ligand_opt = rw.GetMol()

    ligand_coords = get_coordinates(ligand_opt)
    ligand_opt = set_coordinates(ligand_pre_opt, coords=ligand_coords)
    ligand_rmsd = rdMolAlign.CalcRMS(ligand_opt, ligand_pre_opt)
    logger.info(f"Ligand RMSD: {ligand_rmsd:.2f}")

    return ligand_opt, protein_opt
