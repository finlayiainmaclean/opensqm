# ruff: noqa: D100, D103, PLW2901
import tempfile
import time
from pathlib import Path

import numpy as np
from loguru import logger
from rdkit import Chem

from opensqm.mopac.geometry import (
    _finalize_setpi_after_geometry,
    _pi_bonds_prepare_setpi_file,
    rdkit_to_mopac,
)
from opensqm.mopac.lewis import get_cvb_str
from opensqm.mopac.nitro import fix_nitro_groups
from opensqm.mopac.parse_output import _extract_coords
from opensqm.mopac.runner import _run_mopac_input_file, check_mopac_was_success
from opensqm.rdkit_utils import set_coordinates


def run_opt_from_rdmol(
    rdmol: Chem.Mol,
    /,
    *,
    mopac_keywords: list[str],
    charge: int = 0,
    opt_mask: np.ndarray | None = None,
) -> Chem.Mol:
    # MOPAC optimizes the nitro-augmented graph; write coordinates back onto that
    # same mol (atom index order matches GEO_DAT / output table).
    rdmol = fix_nitro_groups(rdmol)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mop_path = tmpdir / "mol.mop"
        setpi_path = tmpdir / "setpi.txt"
        mopac_path = tmpdir / "run.mopac"
        out_path = tmpdir / "run    ac.out"

        pi_bonds = _pi_bonds_prepare_setpi_file(rdmol, setpi_path)
        rdkit_to_mopac(rdmol, mop_path, opt_mask=opt_mask)
        _finalize_setpi_after_geometry(pi_bonds, setpi_path, mopac_keywords=mopac_keywords)
        cvb_str = get_cvb_str(rdmol)

        mopac_keywords.append(f"CHARGE={charge}")
        mopac_keywords.append(f'GEO_DAT="{mop_path!s}"')
        mopac_keywords.append(cvb_str)

        mopac_str = " ".join(mopac_keywords)

        logger.debug(f"Optimisation complex calculation:\n{mopac_str}")
        t0 = time.time()
        mopac_path.write_text(mopac_str)

        _run_mopac_input_file(mopac_path, cwd=tmpdir)
        check_mopac_was_success(out_path.read_text())

        logger.debug(f"Time taken: {time.time() - t0}")

        df = _extract_coords(out_path.read_text())

        if len(df) != rdmol.GetNumAtoms():
            raise ValueError(
                f"MOPAC coordinate rows ({len(df)}) != atoms in mol ({rdmol.GetNumAtoms()})"
            )

        coords = df[["x", "y", "z"]].to_numpy(dtype=float)

        for a, symbol in zip(rdmol.GetAtoms(), df["symbol"], strict=True):
            if a.GetSymbol() != symbol:
                raise ValueError(
                    f"atom index {a.GetIdx()}: mol {a.GetSymbol()} vs MOPAC output {symbol}"
                )

        return set_coordinates(rdmol, coords=coords)
