"""Tests for realigning an MD frame back onto the input protein's frame."""

from pathlib import Path

import numpy as np
import pytest
from openmm import unit
from openmm.app.pdbfile import PDBFile

from opensqm.md.align import align_positions_to_reference, kabsch_rt

# Eight CA-only glycines on chain A plus a calcium ion (residue CA, atom CA) and
# a ligand atom, so the fit has to key on residue identity and skip both
# non-protein residues while still moving them with the protein.
_REFERENCE_PDB = """\
ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00           C
ATOM      3  CA  GLY A   3       7.600   1.000   0.000  1.00  0.00           C
ATOM      4  CA  GLY A   4      11.400   2.500   0.500  1.00  0.00           C
ATOM      5  CA  GLY A   5      15.200   2.000  -1.500  1.00  0.00           C
ATOM      6  CA  GLY A   6      19.000   0.500  -2.000  1.00  0.00           C
ATOM      7  CA  GLY A   7      22.800  -1.000  -1.000  1.00  0.00           C
ATOM      8  CA  GLY A   8      26.600  -0.500   1.000  1.00  0.00           C
HETATM    9 CA    CA A 101      10.000   8.000   0.000  1.00  0.00          CA
END
"""


def _renumber(pdb_text: str, chain: str, offset: int = 0) -> str:
    """Rewrite the chain id and residue numbers, keeping the PDB columns intact."""
    lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            lines.append(f"{line[:21]}{chain}{int(line[22:26]) + offset:>4}{line[26:]}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _rigid_move(coords_nm: np.ndarray) -> np.ndarray:
    """Rotate 37 degrees about z and translate, mimicking solvation plus MD drift."""
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return coords_nm @ rotation.T + np.array([3.1, -2.4, 1.7])


@pytest.fixture
def reference_pdb(tmp_path: Path) -> Path:
    path = tmp_path / "protein_input.pdb"
    path.write_text(_REFERENCE_PDB)
    return path


def test_kabsch_rt_recovers_a_known_transform() -> None:
    """The Kabsch fit inverts a rigid move exactly."""
    rng = np.random.default_rng(0)
    target = rng.normal(size=(12, 3))
    mobile = _rigid_move(target)

    rotation, translation = kabsch_rt(mobile, target)

    assert np.allclose(mobile @ rotation.T + translation, target, atol=1e-9)
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_align_undoes_translation_and_rotation(reference_pdb: Path) -> None:
    """A rigidly moved frame is mapped back onto the reference coordinates."""
    pdb = PDBFile(str(reference_pdb))
    reference_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer))
    moved = _rigid_move(reference_nm)

    aligned = align_positions_to_reference(pdb.topology, moved, reference_pdb)

    # Every atom comes back, including the ion the fit itself ignored.
    assert np.allclose(aligned, reference_nm, atol=1e-6)


def test_align_preserves_internal_geometry(reference_pdb: Path) -> None:
    """Realignment is rigid: no interatomic distance changes."""
    pdb = PDBFile(str(reference_pdb))
    moved = _rigid_move(np.asarray(pdb.positions.value_in_unit(unit.nanometer)))

    aligned = align_positions_to_reference(pdb.topology, moved, reference_pdb)

    before = np.linalg.norm(moved[:, None, :] - moved[None, :, :], axis=-1)
    after = np.linalg.norm(aligned[:, None, :] - aligned[None, :, :], axis=-1)
    assert np.allclose(before, after, atol=1e-9)


def test_align_accepts_quantity_positions(reference_pdb: Path) -> None:
    """Positions may be an OpenMM Quantity; the result is always nm."""
    pdb = PDBFile(str(reference_pdb))
    reference_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer))

    aligned = align_positions_to_reference(pdb.topology, pdb.positions, reference_pdb)

    assert np.allclose(aligned, reference_nm, atol=1e-6)


def test_align_matches_renamed_and_extended_residues(tmp_path: Path) -> None:
    """Preparation renames titratable residues and adds atoms; the fit still matches.

    The mobile structure stands in for a prepared complex: a HIS renamed to HIE,
    an extra residue PDBFixer built in, and a ligand residue first in the file -
    none of which exist in (or line up positionally with) the input protein.
    """
    reference = tmp_path / "input.pdb"
    reference.write_text(
        _REFERENCE_PDB.replace("GLY A   4", "HIS A   4").replace(
            "ATOM      8  CA  GLY A   8      26.600  -0.500   1.000", ""
        )
    )
    mobile_pdb = tmp_path / "prepared.pdb"
    mobile_pdb.write_text(
        "HETATM    1  C1  LIG B 500       5.000   5.000   5.000  1.00  0.00           C\n"
        + _REFERENCE_PDB.replace("GLY A   4", "HIE A   4")
    )

    pdb = PDBFile(str(mobile_pdb))
    mobile_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer))
    moved = _rigid_move(mobile_nm)

    aligned = align_positions_to_reference(pdb.topology, moved, reference)

    assert np.allclose(aligned, mobile_nm, atol=1e-6)


def test_align_falls_back_to_chain_agnostic_matching(tmp_path: Path) -> None:
    """A reference whose chain id was not preserved still aligns on residue number."""
    reference = tmp_path / "input.pdb"
    reference.write_text(_renumber(_REFERENCE_PDB, "B"))
    mobile_pdb = tmp_path / "prepared.pdb"
    mobile_pdb.write_text(_REFERENCE_PDB)

    pdb = PDBFile(str(mobile_pdb))
    reference_nm = np.asarray(PDBFile(str(reference)).positions.value_in_unit(unit.nanometer))
    moved = _rigid_move(np.asarray(pdb.positions.value_in_unit(unit.nanometer)))

    aligned = align_positions_to_reference(pdb.topology, moved, reference)

    assert np.allclose(aligned, reference_nm, atol=1e-6)


def test_align_leaves_frame_alone_when_nothing_matches(tmp_path: Path) -> None:
    """An unmatchable reference is a no-op, not a failure late in a paid-for run."""
    reference = tmp_path / "other.pdb"
    reference.write_text(_renumber(_REFERENCE_PDB, "Z", offset=500))
    mobile_pdb = tmp_path / "prepared.pdb"
    mobile_pdb.write_text(_REFERENCE_PDB)

    pdb = PDBFile(str(mobile_pdb))
    moved = _rigid_move(np.asarray(pdb.positions.value_in_unit(unit.nanometer)))

    aligned = align_positions_to_reference(pdb.topology, moved, reference)

    assert np.allclose(aligned, moved)


def test_align_leaves_frame_alone_when_reference_is_unreadable(tmp_path: Path) -> None:
    """A reference that is not a readable PDB is a no-op too."""
    reference = tmp_path / "input.cif"
    reference.write_text("data_notapdb\n")
    mobile_pdb = tmp_path / "prepared.pdb"
    mobile_pdb.write_text(_REFERENCE_PDB)

    pdb = PDBFile(str(mobile_pdb))
    moved = _rigid_move(np.asarray(pdb.positions.value_in_unit(unit.nanometer)))

    assert np.allclose(align_positions_to_reference(pdb.topology, moved, reference), moved)
