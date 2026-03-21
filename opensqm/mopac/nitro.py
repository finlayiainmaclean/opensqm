# ruff: noqa: D100, D401
from rdkit import Chem

# Marks O-O bonds added only for MOPAC/LEWIS (`fix_nitro_groups`).
MOPAC_NITRO_AUX_BOND = "mopac_nitro_aux_bond"


def strip_mopac_nitro_aux_bonds(mol: Chem.Mol) -> Chem.Mol:
    """Remove auxiliary nitro O-O bonds, then sanitize (restore standard nitro valences/charges)."""
    rw = Chem.RWMol(Chem.Mol(mol))
    pairs = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in rw.GetBonds()
        if b.HasProp(MOPAC_NITRO_AUX_BOND) and b.GetProp(MOPAC_NITRO_AUX_BOND) == "1"
    ]
    for a, b in pairs:
        rw.RemoveBond(int(a), int(b))

    # `fix_nitro_groups` zeroed formal charges; restore [N+](=O)[O-] before sanitize.
    # Each aux bond is O-O; after removal each O has only the nitro N as neighbor.
    for a, b in pairs:
        na = [n.GetIdx() for n in rw.GetAtomWithIdx(a).GetNeighbors()]
        nb = [n.GetIdx() for n in rw.GetAtomWithIdx(b).GetNeighbors()]
        if len(na) != 1 or len(nb) != 1 or na[0] != nb[0]:
            raise ValueError("expected nitro O-O aux bond with shared N neighbor after strip")
        n_idx = na[0]
        rw.GetAtomWithIdx(n_idx).SetFormalCharge(1)
        rw.GetAtomWithIdx(a).SetFormalCharge(-1)
        rw.GetAtomWithIdx(b).SetFormalCharge(0)

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def fix_nitro_groups(mol: Chem.Mol) -> Chem.Mol:
    """
    Finds nitro groups in a molecule and adds single bonds between the two oxygens.

    See: https://github.com/openmopac/mopac/blob/main/src/chemistry/lewis.F90

    Args:
        mol: RDKit Mol object

    Returns
    -------
        RDKit Mol object with bonds added between nitro group oxygens
    """
    # Make a copy to avoid modifying the original molecule
    mol = Chem.Mol(mol)

    # Define SMARTS pattern for nitro group: [N+](=O)[O-]
    # This matches N with +1, double bonded to O, single bonded to O with -1
    nitro_pattern = Chem.MolFromSmarts("[N;D3](~[O;D1])(~[O;D1])")

    if nitro_pattern is None:
        return mol

    matches = mol.GetSubstructMatches(nitro_pattern)
    if not matches:
        return mol

    editable_mol = Chem.EditableMol(mol)
    bonds_added = set()

    for match in matches:
        n_idx, o1_idx, o2_idx = match

        if mol.GetAtomWithIdx(o1_idx).GetDegree() != 1:
            continue
        if mol.GetAtomWithIdx(o2_idx).GetDegree() != 1:
            continue

        bond_key = tuple(sorted([o1_idx, o2_idx]))
        if bond_key not in bonds_added:
            if mol.GetBondBetweenAtoms(o1_idx, o2_idx) is None:
                editable_mol.AddBond(o1_idx, o2_idx, Chem.BondType.SINGLE)
                bonds_added.add(bond_key)

    # Build modified mol from editable version
    modified_mol = editable_mol.GetMol()

    # Now zero charges on atoms in modified mol
    for match in matches:
        n_idx, o1_idx, o2_idx = match
        for idx in (n_idx, o1_idx, o2_idx):
            modified_mol.GetAtomWithIdx(idx).SetFormalCharge(0)

    for o1_idx, o2_idx in bonds_added:
        bond = modified_mol.GetBondBetweenAtoms(o1_idx, o2_idx)
        if bond is not None:
            bond.SetProp(MOPAC_NITRO_AUX_BOND, "1")

    return modified_mol
