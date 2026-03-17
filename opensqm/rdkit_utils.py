"""RDKit utilities."""

from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdsl import select_atom_ids, select_molecule


def set_residue_info(
    mol: Chem.Mol, /, *, chain_id: str = "Z", resname: str = "LIG", resnum: int = 999
) -> Chem.Mol:
    """Mutate `mol` in-place so that all atoms belong to the same PDB residue.

    RDKit rules / caveats
    ---------------------
    • Each Atom carries an AtomPDBResidueInfo object; create one if missing.
    • `chain_id` must be a single character (PDB allows only 1).
    • `resnum` must be 1-9999 to stay within PDB formatting limits.
    """
    # Roundtrip to PDB to set the PDB residue info.
    mol_copy = Chem.MolFromPDBBlock(Chem.MolToPDBBlock(mol), sanitize=False, removeHs=False)

    if len(chain_id) != 1:
        raise ValueError("PDB chain IDs are exactly one character")

    for at, at_copy in zip(mol.GetAtoms(), mol_copy.GetAtoms(), strict=False):
        info = at_copy.GetPDBResidueInfo()
        info.SetChainId(str(chain_id))
        info.SetResidueNumber(int(resnum))
        info.SetResidueName(str(resname))
        at.SetMonomerInfo(info)

    return mol


def get_coordinates(mol: Chem.Mol, /, *, conf_id: int = 0) -> np.ndarray:
    """Get coordinates from conformer."""
    coords = mol.GetConformer(conf_id).GetPositions()
    return coords


def set_random_coordinates(mol: Chem.Mol, /):
    """Set random coordinates."""
    ps = AllChem.ETKDGv3()  # type: ignore
    ps.useRandomCoords = True
    AllChem.EmbedMolecule(mol, ps)  # type: ignore


def set_coordinates(mol: Chem.Mol, /, *, coords: np.ndarray, conf_id: int = 0) -> Chem.Mol:
    """Overwrite (or create) conformer `conf_id` with Cartesian coordinates.

    `coords` must be shape (N_atoms, 3).
    Returns `mol` for convenience.
    """
    mol = Chem.Mol(mol)

    n_atoms = mol.GetNumAtoms()
    if coords.shape != (n_atoms, 3):
        raise ValueError(f"coords shape {coords.shape} does not match atom count {n_atoms}")

    # make sure the conformer exists
    if mol.GetNumConformers() == 0:
        set_random_coordinates(mol)

    try:
        conf = mol.GetConformer(conf_id)
    except ValueError:
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetId(conf_id)
        mol.AddConformer(conf, assignId=False)

    # write xyz into the conformer
    for idx, (x, y, z) in enumerate(coords.astype(float)):
        conf.SetAtomPosition(idx, Point3D(x, y, z))

    return mol


def submol(mol: Chem.Mol, /, *, atom_ids: Sequence[int]) -> Chem.Mol:
    mol = Chem.RWMol(Chem.Mol(mol))
    to_delete_set = set(range(mol.GetNumAtoms())) - set(atom_ids)

    for atom in mol.GetAtoms():
        atom.SetIntProp("_original_idx", atom.GetIdx())

    for idx in sorted(to_delete_set, reverse=True):
        mol.RemoveAtom(idx)

    mol = mol.GetMol()

    Chem.SanitizeMol(mol)

    return mol


def refine_caps(mol: Chem.Mol, ace_res_ids: list, nme_res_ids: list):
    rw_mol = Chem.RWMol(mol)
    atoms_to_remove = []
    cap_target_indices = []

    # 1. Identify what to keep and what to kill
    for atom in rw_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if not info:
            continue

        res_id = (info.GetChainId(), info.GetResidueNumber())
        atom_name = info.GetName().strip()

        if res_id in ace_res_ids:
            if atom_name not in ["CA", "C", "O"]:
                atoms_to_remove.append(atom.GetIdx())
            elif atom_name == "CA":
                # This CA needs 3 hydrogens to become a Methyl
                atom.SetProp("is_cap_terminal", "true")
                info.SetResidueName("ACE")
            elif atom_name in ["C", "O"]:
                info.SetResidueName("ACE")

        elif res_id in nme_res_ids:
            if atom_name not in ["N", "CA"]:
                atoms_to_remove.append(atom.GetIdx())
            elif atom_name == "CA":
                # This CA needs 3 hydrogens to become a Methyl
                atom.SetProp("is_cap_terminal", "true")

                info.SetResidueName("NME")
            elif atom_name == "N":
                # This N needs 1 hydrogen (the amide H)
                atom.SetProp("is_cap_terminal", "true")

                info.SetResidueName("NME")

        atom.SetPDBResidueInfo(info)

    # 2. Perform deletion
    for idx in sorted(atoms_to_remove, reverse=True):
        rw_mol.RemoveAtom(idx)

    # 3. Find the new indices of the tagged atoms
    final_mol = rw_mol.GetMol()
    for atom in final_mol.GetAtoms():
        if atom.HasProp("is_cap_terminal"):
            cap_target_indices.append(atom.GetIdx())

    return final_mol, cap_target_indices


def crop_and_cap_protein(
    *, protein: Chem.Mol, ligand: Chem.Mol, distance_to_ligand: int = 5
) -> Chem.Mol:
    # 1. Selection logic (same as before)

    complex = Chem.CombineMols(protein, ligand)
    selection = select_molecule(complex, f"byres (protein within {distance_to_ligand} of resn LIG)")

    pocket = selection.mol
    if pocket is None:
        raise ValueError("No pocket found")

    pocket_residues = set()
    for atom in selection.mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        pocket_residues.add((info.GetChainId(), info.GetResidueNumber()))

    ace_res_ids = []  # Store as (chain, res_num)
    nme_res_ids = []

    # Identify which residues will become our caps
    for chain, res_num in pocket_residues:
        prev = (chain, res_num - 1)
        if prev not in pocket_residues:
            ace_res_ids.append(prev)

        nxt = (chain, res_num + 1)
        if nxt not in pocket_residues:
            nme_res_ids.append(nxt)

    # Get *all* atom IDs for these residues (Backbone + Sidechains)
    # We need the whole residue first so refine_caps can prune it correctly
    keep_ids = list(selection.atom_mapping.values())
    for res in ace_res_ids + nme_res_ids:
        ids = select_atom_ids(complex, f"(chain '{res[0]}') and (resi {res[1]})")
        keep_ids.extend([int(i) for i in ids])

    # Extract the submolecule
    capped_protein = submol(complex, atom_ids=keep_ids)

    # Mutate the extra residues into ACE/NME caps
    # This removes sidechains and renames residues
    capped_protein, cap_ids = refine_caps(capped_protein, ace_res_ids, nme_res_ids)

    for atom in capped_protein.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info and info.GetResidueName() == "ACE" and info.GetName() == "CA":
            atom.SetExplicitValence(3)
            atom.SetImplicitValence(0)

    for atom in capped_protein.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info and info.GetResidueName() == "NME" and info.GetName() == "CA":
            atom.SetExplicitValence(3)
            atom.SetImplicitValence(0)

        if info and info.GetResidueName() == "NME" and info.GetName() == "N":
            atom.SetExplicitValence(1)
            atom.SetImplicitValence(0)

    # Chem.SanitizeMol(capped_protein)

    # Finalize with hydrogens
    # AddHs will now see a methyl group and add 3 hydrogens to the CA
    capped_protein = Chem.AddHs(capped_protein, addCoords=True, onlyOnAtoms=cap_ids)

    return capped_protein


if __name__ == "__main__":
    protein = Chem.MolFromPDBFile(
        "data/inputs/PL-REX/003-CK2/1ZOH.prot.pdb", removeHs=False, sanitize=False
    )
    ligand = Chem.MolFromMolFile("data/inputs/PL-REX/003-CK2/1ZOH.sdf", removeHs=False)

    ligand = set_residue_info(ligand)
    capped_protein = crop_and_cap_protein(protein=protein, ligand=ligand, distance_to_ligand=10)
    Chem.MolToPDBFile(capped_protein, "/tmp/capped_protein.pdb")
