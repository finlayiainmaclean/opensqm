# ruff: noqa: D100, D103, PLW2901
import tempfile
from pathlib import Path

from rdkit import Chem

from opensqm.mopac.geometry import (
    _finalize_setpi_after_geometry,
    _pi_bonds_prepare_setpi_file,
    rdkit_to_mopac,
)
from opensqm.mopac.lewis import get_cvb_str
from opensqm.mopac.nitro import fix_nitro_groups
from opensqm.mopac.parse_output import _extract_formal_charges_from_mopac_str
from opensqm.mopac.runner import _run_mopac_input_file, check_mopac_was_success


def get_mopac_formal_charges(rdmol: Chem.Mol) -> tuple[int, dict[int, int]]:
    rdmol = fix_nitro_groups(rdmol)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mol_mop_path = tmpdir / "mol.mop"
        mopac_path = tmpdir / "run.mopac"
        out_path = tmpdir / "run    ac.out"
        setpi_path = tmpdir / "setpi.txt"

        pi_bonds = _pi_bonds_prepare_setpi_file(rdmol, setpi_path)
        rdkit_to_mopac(rdmol, mol_mop_path)

        cvb_str = get_cvb_str(rdmol)

        mopac_str = f'CHARGES LET GEO_DAT="{mol_mop_path!s}" {cvb_str}'
        mopac_str += _finalize_setpi_after_geometry(pi_bonds, setpi_path)

        mopac_path.write_text(mopac_str)
        _run_mopac_input_file(mopac_path, cwd=tmpdir)
        output_str = out_path.read_text()
        check_mopac_was_success(output_str)

        charge_df = _extract_formal_charges_from_mopac_str(output_str)

        if len(charge_df) == 0:
            charge = 0
            formal_charges = {}
        else:
            charge = charge_df.charge.sum()

            formal_charges = dict(zip(charge_df.atom_ix, charge_df.charge, strict=False))

            formal_charges = {k - 1: v for k, v in formal_charges.items()}

    return charge, formal_charges
