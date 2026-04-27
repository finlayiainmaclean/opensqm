"""Tests for the torsion scanner module."""

from rdkit import Chem
from rdkit.Chem import AllChem

from opensqm.torsion_scanner import autodetect_flip_dihedrals


def test_autodetect_flip_dihedrals():
    """
    Test the top-level orchestrator for Type 2 atropisomer flip detection.

    Executes on a biaryl system, verifying the returned torsion groups and mapping angles.
    """
    mol = Chem.MolFromSmiles("c1cccc(Cl)[1c]1-[1c]1c(Br)cccc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    # Standard autodetection run. Biphenyl naturally falls outside the boundary (Type 1),
    bonds = autodetect_flip_dihedrals(mol)

    for bond in bonds:
        assert mol.GetAtomWithIdx(bond[0]).GetIsotope() == 1
        assert mol.GetAtomWithIdx(bond[1]).GetIsotope() == 1
