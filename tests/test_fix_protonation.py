"""Tests for PROPKA/PDB2PQR protonation in :mod:`opensqm.fix`."""

from pathlib import Path

import pytest

from opensqm.fix import PDB2PQRError, _protonate_with_propka, run_pdbfixer

# Solvated ACE-ASP-NME capped dipeptide, checked into the repo for constant-pH
# reference-energy generation (opensqm/cph/model-compounds) - reused here as a small,
# always-available structure with a real ASP.
_ASP_PDB = Path(__file__).resolve().parents[1] / "opensqm" / "cph" / "model-compounds" / "ASP.pdb"


def _with_protonated_asp(dest: Path) -> Path:
    """Write the ASP model compound to *dest* with an ASH-style HD2 on its ASP.

    HD2 lives only in PDB2PQR's ASH patch, never in plain ASP, so PDB2PQR bonds it to
    nothing and aborts debumping with "Found gap in biomolecule structure". PDB2PQR
    writes exactly this hydrogen whenever PROPKA calls an ASP protonated at the
    requested pH, so its own output used to be un-preparable.
    """
    lines = _ASP_PDB.read_text().splitlines(keepends=True)
    od2 = next(
        i
        for i, line in enumerate(lines)
        if line.startswith(("ATOM", "HETATM")) and line[12:16] == " OD2" and line[17:20] == "ASP"
    )
    # An O-H away from OD2, i.e. what PDB2PQR itself writes for a protonated ASP.
    x = float(lines[od2][30:38]) + 0.97
    lines.insert(
        od2 + 1,
        f"{lines[od2][:12]} HD2{lines[od2][16:30]}{x:8.3f}{lines[od2][38:76]} H  \n",
    )
    dest.write_text("".join(lines))
    return dest


def test_run_pdbfixer_tolerates_hydrogens_pdb2pqr_did_not_place(tmp_path: Path) -> None:
    """Preparing an already-prepared structure must work, protonated ASP and all."""
    out, pka_groups = run_pdbfixer(
        _with_protonated_asp(tmp_path / "protonated_asp.pdb"),
        tmp_path / "out.pdb",
        ph=7.4,
    )
    text = Path(out).read_text()
    # The input HD2 is dropped and the titration state re-decided from scratch: a
    # solvated ASP has a pKa near 3.8, so at pH 7.4 it comes back deprotonated.
    assert " HD2 ASP" not in text
    assert " HB2 ASP" in text
    assert any(g["res_name"] == "ASP" and g["group_type"] == "COO" for g in pka_groups)


def test_protonate_with_propka_reports_why_pdb2pqr_failed(tmp_path: Path) -> None:
    """PDB2PQR re-raises a bare, empty RuntimeError; the reason must not be lost."""
    broken = _with_protonated_asp(tmp_path / "broken.pdb")
    with pytest.raises(PDB2PQRError, match=r"Found gap in biomolecule structure.*HD2 ASP"):
        _protonate_with_propka(broken, tmp_path / "out.pdb", 7.4)
