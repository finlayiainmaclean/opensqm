"""The MMGBSA representative outputs land in the input protein's frame.

Exercises the wiring rather than the fit itself (see ``test_align``): a stand-in
"MD frame" - protein, ligand and waters, rigidly moved away from the input as
solvation and MD would move it - is pushed through the writers of both the
explicit (``run_mmgbsa._write_representative``) and implicit
(``run_mmgbsa_implicit._write_outputs``) paths, and the published PDB/SDF pair
is checked back against the input protein.
"""

from pathlib import Path

import numpy as np
import pytest
from openmm import Vec3, unit
from openmm.app.pdbfile import PDBFile
from rdkit import Chem

from opensqm.md.run_mmgbsa import MMGBSASettings, _write_representative
from opensqm.md.run_mmgbsa_implicit import _write_outputs

# A minimal complex in the layout the pipelines produce: the ligand first, then
# the protein, then waters (two near the ligand, one far away) - enough for the
# n-closest-waters trim to have something to drop.
_COMPLEX_PDB = """\
HETATM    1  C1  LIG A 900       9.500   4.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG A 900      10.700   4.400   0.000  1.00  0.00           C
ATOM      3  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      4  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00           C
ATOM      5  CA  GLY A   3       7.600   1.000   0.000  1.00  0.00           C
ATOM      6  CA  HIE A   4      11.400   2.500   0.500  1.00  0.00           C
ATOM      7  CA  GLY A   5      15.200   2.000  -1.500  1.00  0.00           C
ATOM      8  CA  GLY A   6      19.000   0.500  -2.000  1.00  0.00           C
ATOM      9  CA  GLY A   7      22.800  -1.000  -1.000  1.00  0.00           C
ATOM     10  CA  GLY A   8      26.600  -0.500   1.000  1.00  0.00           C
HETATM   11  O   HOH A 601      10.000   6.500   0.000  1.00  0.00           O
HETATM   12  O   HOH A 602       8.000   6.000   1.000  1.00  0.00           O
HETATM   13  O   HOH A 603      40.000  40.000  40.000  1.00  0.00           O
END
"""

# The same protein as the input file: no ligand, no waters, and the titratable
# residue still under its input name (preparation renames HIS -> HIE).
_INPUT_PROTEIN_PDB = (
    "\n".join(
        line
        for line in _COMPLEX_PDB.replace("HIE A   4", "HIS A   4").splitlines()
        if " CA " in line and "LIG" not in line
    )
    + "\nEND\n"
)

_LIGAND_RESNAME = "LIG"


def _rigid_move(coords_nm: np.ndarray) -> np.ndarray:
    """Rotate 52 degrees about y and translate, as solvation plus MD would."""
    angle = np.deg2rad(52.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    return coords_nm @ rotation.T + np.array([12.0, -7.5, 4.25])


@pytest.fixture
def frame(tmp_path: Path) -> tuple[Path, PDBFile, np.ndarray]:
    """``(input protein path, complex PDB, moved coordinates nm)``."""
    input_protein = tmp_path / "protein_input.pdb"
    input_protein.write_text(_INPUT_PROTEIN_PDB)
    complex_path = tmp_path / "complex.pdb"
    complex_path.write_text(_COMPLEX_PDB)
    complex_pdb = PDBFile(str(complex_path))
    moved = _rigid_move(np.asarray(complex_pdb.positions.value_in_unit(unit.nanometer)))
    return input_protein, complex_pdb, moved


def _two_carbon_mol() -> Chem.Mol:
    """A two-carbon stand-in for the winning protomer, in the frame's atom order."""
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(6))
    mol.AddBond(0, 1, Chem.BondType.SINGLE)
    out = mol.GetMol()
    conf = Chem.Conformer(2)
    out.AddConformer(conf, assignId=True)
    return out


def _input_calpha_angstrom(input_protein: Path) -> np.ndarray:
    pdb = PDBFile(str(input_protein))
    coords = np.asarray(pdb.positions.value_in_unit(unit.angstrom))
    return np.array([coords[a.index] for a in pdb.topology.atoms() if a.name == "CA"])


def _written_calpha_angstrom(prot_path: Path) -> np.ndarray:
    pdb = PDBFile(str(prot_path))
    coords = np.asarray(pdb.positions.value_in_unit(unit.angstrom))
    return np.array([coords[a.index] for a in pdb.topology.atoms() if a.name == "CA"])


def test_explicit_representative_is_written_in_the_input_frame(
    frame: tuple[Path, PDBFile, np.ndarray], tmp_path: Path
) -> None:
    """The explicit path's prot.pdb/lig.sdf come back onto the input protein."""
    input_protein, complex_pdb, moved = frame
    prot_path, lig_path = tmp_path / "prot.pdb", tmp_path / "lig.sdf"
    positions = unit.Quantity([Vec3(*row) for row in moved], unit.nanometer)

    _write_representative(
        complex_pdb.topology,
        positions,
        _two_carbon_mol(),
        MMGBSASettings(n_closest_waters=2, ligand_resname=_LIGAND_RESNAME),
        prot_path,
        lig_path,
        reference_protein=input_protein,
    )

    assert np.allclose(
        _written_calpha_angstrom(prot_path), _input_calpha_angstrom(input_protein), atol=1e-3
    )
    # The ligand travels with the protein: back at its input pose beside residue 3/4.
    ligand = Chem.MolFromMolFile(str(lig_path), removeHs=False)
    ligand_coords = np.array(ligand.GetConformer().GetPositions())
    assert np.allclose(ligand_coords[0], [9.5, 4.0, 0.0], atol=1e-3)
    assert np.allclose(ligand_coords[1], [10.7, 4.4, 0.0], atol=1e-3)
    # Trimming still ran: the distant water was dropped, the two near ones kept.
    waters = [r for r in PDBFile(str(prot_path)).topology.residues() if r.name == "HOH"]
    assert len(waters) == 2


def test_explicit_representative_without_reference_stays_in_the_md_frame(
    frame: tuple[Path, PDBFile, np.ndarray], tmp_path: Path
) -> None:
    """Without a reference the frame is written as-is (the pre-realignment behaviour)."""
    _input_protein, complex_pdb, moved = frame
    prot_path, lig_path = tmp_path / "prot.pdb", tmp_path / "lig.sdf"

    _write_representative(
        complex_pdb.topology,
        unit.Quantity([Vec3(*row) for row in moved], unit.nanometer),
        _two_carbon_mol(),
        MMGBSASettings(n_closest_waters=2, ligand_resname=_LIGAND_RESNAME),
        prot_path,
        lig_path,
    )

    calpha_moved = np.array(
        [
            moved[a.index] * 10.0
            for a in complex_pdb.topology.atoms()
            if a.name == "CA" and a.residue.name != _LIGAND_RESNAME
        ]
    )
    assert np.allclose(_written_calpha_angstrom(prot_path), calpha_moved, atol=1e-3)


def test_implicit_outputs_are_written_in_the_input_frame(
    frame: tuple[Path, PDBFile, np.ndarray], tmp_path: Path
) -> None:
    """The implicit path's prot.pdb/lig.sdf come back onto the input protein too."""
    input_protein, complex_pdb, moved = frame
    prot_path, lig_path = tmp_path / "prot.pdb", tmp_path / "lig.sdf"

    _write_outputs(
        complex_pdb.topology,
        moved,
        _two_carbon_mol(),
        _LIGAND_RESNAME,
        prot_path,
        lig_path,
        reference_protein=input_protein,
    )

    assert np.allclose(
        _written_calpha_angstrom(prot_path), _input_calpha_angstrom(input_protein), atol=1e-3
    )
    ligand_coords = np.array(
        Chem.MolFromMolFile(str(lig_path), removeHs=False).GetConformer().GetPositions()
    )
    assert np.allclose(ligand_coords[0], [9.5, 4.0, 0.0], atol=1e-3)
