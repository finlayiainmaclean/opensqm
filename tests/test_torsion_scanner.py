"""Tests for the torsion scanner module."""

from rdkit import Chem
from rdkit.Chem import AllChem

from opensqm.torsion_scanner import autodetect_flip_dihedrals


def test_autodetect_flip_dihedrals():
    """
    Test the top-level orchestrator for Type 2 atropisomer flip detection.

    1-phenylnaphthalene's biaryl bond is a well-known moderate-barrier atropisomer
    (naphthalene peri-strain hinders rotation without fully blocking it): the relaxed
    torsion scan puts it around 17 kcal/mol, comfortably inside the
    [MIN_TS_ENERGY, MAX_TS_ENERGY] = [5, 26.5] kcal/mol window (itself derived from
    ``is_type_2_atropisomer``'s 1ns-1month half-life bounds via the Eyring equation) with
    margin on both sides. An earlier, more sterically hindered ortho,ortho'-dihalobiphenyl
    sat right at that upper boundary (~26-32 kcal/mol depending on the exact SMIRNOFF
    version), making this test intermittently fail on unrelated dependency updates.
    """
    mol = Chem.MolFromSmiles("c1ccc(-c2cccc3ccccc23)cc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    bonds = autodetect_flip_dihedrals(mol)

    assert len(bonds) == 1

    ring_info = mol.GetRingInfo()
    for bond in bonds:
        atom_a, atom_b = mol.GetAtomWithIdx(bond[0]), mol.GetAtomWithIdx(bond[1])
        assert atom_a.GetIsAromatic()
        assert atom_b.GetIsAromatic()
        # The flip bond joins the phenyl and naphthalene ring systems - its two
        # atoms should not share a single ring with each other.
        assert not any(bond[0] in ring and bond[1] in ring for ring in ring_info.AtomRings())
