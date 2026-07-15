"""Tests for opensqm.cph.reference_energy.get_hydrogen_variants.

The variants returned by `get_hydrogen_variants` are the ones we hand to
`Modeller.addHydrogens(variants=...)` when building a `ConstantPH` from a
custom ligand. To validate them we round-trip:

    rdkit ligand
        -> OpenMM topology + positions (with H)
        -> strip H
        -> addHydrogens(variants=get_hydrogen_variants(topology))
        -> write PDB
        -> reload via rdkit/OpenFF using the original SMILES as a template
        -> canonical SMILES must match the input.
"""

import tempfile
from pathlib import Path

import pytest
from openff.toolkit.topology import Molecule, Topology  # type: ignore
from openmm import unit  # type: ignore
from openmm.app import Modeller, PDBFile  # type: ignore
from rdkit import Chem

from opensqm.cph.reference_energy import (
    build_imidazole_states,
    get_hydrogen_variants,
)


def _round_trip_via_get_hydrogen_variants(rdkit_mol: Chem.Mol) -> str:
    """Apply the get_hydrogen_variants -> addHydrogens round-trip and return the SMILES."""
    expected_smiles = Chem.MolToSmiles(Chem.RemoveHs(rdkit_mol))

    offmol = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=False)
    top = offmol.to_topology().to_openmm()
    pos = (offmol.conformers[0].m * unit.angstrom).in_units_of(unit.nanometer)

    variants = get_hydrogen_variants(top)

    modeller = Modeller(top, pos)
    modeller.delete([a for a in modeller.topology.atoms() if a.element.symbol == "H"])
    modeller.addHydrogens(variants=variants)

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
        pdb_path = f.name

    try:
        # OpenFF uses the SMILES as a template to recover bond orders that the PDB drops.
        template = Molecule.from_smiles(expected_smiles)
        roundtrip_top = Topology.from_pdb(pdb_path, unique_molecules=[template])
    finally:
        Path(pdb_path).unlink(missing_ok=True)

    roundtrip_mol = next(roundtrip_top.molecules)
    # Canonicalise via rdkit so we're comparing the same flavour of SMILES.
    return Chem.MolToSmiles(Chem.MolFromSmiles(roundtrip_mol.to_smiles(explicit_hydrogens=False)))


def test_get_hydrogen_variants_returns_one_entry_per_residue() -> None:
    """`get_hydrogen_variants` should return one entry per residue in the topology."""
    imidazolium, _ = build_imidazole_states()
    offmol = Molecule.from_rdkit(imidazolium, allow_undefined_stereo=False)
    top = offmol.to_topology().to_openmm()

    variants = get_hydrogen_variants(top)

    assert len(variants) == top.getNumResidues() == 1


def test_get_hydrogen_variants_imidazole_round_trip() -> None:
    """Re-adding hydrogens via the variants must reproduce the input SMILES."""
    _, imidazole = build_imidazole_states()
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RemoveHs(imidazole))))

    got = _round_trip_via_get_hydrogen_variants(imidazole)

    assert got == expected, f"round-trip SMILES {got!r} != expected {expected!r}"


def test_get_hydrogen_variants_imidazolium_round_trip() -> None:
    """Same round trip but for the protonated (+1) state."""
    imidazolium, _ = build_imidazole_states()
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RemoveHs(imidazolium))))

    got = _round_trip_via_get_hydrogen_variants(imidazolium)

    assert got == expected, f"round-trip SMILES {got!r} != expected {expected!r}"


def test_get_hydrogen_variants_h_attachments_match_topology() -> None:
    """Each (h_name, parent_name) tuple must reference real atoms in the topology and the
    parent must actually be bonded to that hydrogen.
    """
    imidazolium, _ = build_imidazole_states()
    offmol = Molecule.from_rdkit(imidazolium, allow_undefined_stereo=False)
    top = offmol.to_topology().to_openmm()

    variants = get_hydrogen_variants(top)
    assert len(variants) == 1
    residue_variant = variants[0]

    atoms_by_name = {a.name: a for a in next(top.residues()).atoms()}
    bonded_pairs: set[frozenset[str]] = set()
    for bond in top.bonds():
        a1, a2 = bond[0], bond[1]
        if a1.residue == a2.residue:
            bonded_pairs.add(frozenset({a1.name, a2.name}))

    for h_name, parent_name in residue_variant:
        assert h_name in atoms_by_name, f"hydrogen {h_name!r} not in topology"
        assert parent_name in atoms_by_name, f"parent {parent_name!r} not in topology"
        assert atoms_by_name[h_name].element.symbol == "H"
        assert atoms_by_name[parent_name].element.symbol != "H"
        assert frozenset({h_name, parent_name}) in bonded_pairs, (
            f"({h_name}, {parent_name}) is not a real bond in the topology"
        )

    # Imidazolium has 5 hydrogens (3 C-H + 2 N-H).
    assert len(residue_variant) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
