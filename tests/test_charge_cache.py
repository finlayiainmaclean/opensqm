"""Tests for run-local ligand variant cache."""

import numpy as np
from openff.toolkit.topology import Molecule  # type: ignore
from openff.units import unit

from opensqm.md.charge_cache import load_ligand_variant, save_ligand_variant


def test_run_local_ligand_variant_roundtrip(tmp_path) -> None:
    mol = Molecule.from_smiles("CCO", allow_undefined_stereo=True)
    mol.name = "LIG"
    mol.generate_conformers(n_conformers=1)
    mol.partial_charges = np.zeros(mol.n_atoms) * unit.elementary_charge

    save_ligand_variant(tmp_path, mol)
    loaded = load_ligand_variant(tmp_path, "LIG")

    assert loaded is not None
    assert loaded.name == "LIG"
    assert loaded.partial_charges is not None
    np.testing.assert_allclose(
        loaded.partial_charges.m_as(unit.elementary_charge),
        mol.partial_charges.m_as(unit.elementary_charge),
    )
