# ruff: noqa: D100, D103, PLW2901
import tempfile
import time
from pathlib import Path
from typing import Literal

from loguru import logger
from rdkit import Chem

from opensqm.mopac.exceptions import MOPACError
from opensqm.mopac.geometry import (
    _finalize_setpi_after_geometry,
    _pi_bonds_prepare_setpi_file,
    rdkit_to_mopac,
)
from opensqm.mopac.lewis import get_cvb_str
from opensqm.mopac.nitro import fix_nitro_groups
from opensqm.mopac.parse_output import _extract_energy, calculate_nonpolar_term
from opensqm.mopac.runner import _run_mopac_input_file, check_mopac_was_success


def run_singlepoint_from_rdmol(
    rdmol: Chem.Mol,
    /,
    *,
    use_mozyme: bool = True,
    solvent: Literal["cosmo", "cosmo2"] | None = "cosmo2",
    charge: int = 0,
) -> float:
    rdmol = fix_nitro_groups(rdmol)

    mopac_keywords = [
        "PM6-D3H4X",
        "NOMM",
        "1SCF",
        "RHF",
        "CUTOFP=10.0",
        "METAL",
        "PRECISE",
        #   "LET",
    ]

    if use_mozyme:
        mopac_keywords.append("MOZYME")

    match solvent:
        case "cosmo":
            mopac_keywords.append("EPS=78.5")
        case "cosmo2":
            mopac_keywords.extend(
                [
                    "EPS=78.5",
                    "VDW(C=1.821;O=1.682;H=0.828;N=1.904;P=2.118;S=2.369;F=1.602;Cl=1.911;Br=2.178;I=2.276)",
                ]
            )

    mopac_keywords = list(mopac_keywords)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mop_path = tmpdir / "mol.mop"
        setpi_path = tmpdir / "setpi.txt"
        mopac_path = tmpdir / "run.mopac"
        out_path = tmpdir / "run    ac.out"

        pi_bonds = _pi_bonds_prepare_setpi_file(rdmol, setpi_path)
        rdkit_to_mopac(rdmol, mop_path)

        cvb_str = get_cvb_str(rdmol)

        mopac_keywords.append(f"CHARGE={charge}")
        mopac_keywords.append(f'GEO_DAT="{mop_path!s}"')
        mopac_keywords.append(cvb_str)
        _finalize_setpi_after_geometry(pi_bonds, setpi_path, mopac_keywords=mopac_keywords)

        mopac_str = " ".join(mopac_keywords)

        mopac_path.write_text(mopac_str)
        logger.debug(f"Singlepoint ligand calculation:\n{mopac_str}")
        t0 = time.time()

        _run_mopac_input_file(mopac_path, cwd=tmpdir)
        output_str = out_path.read_text()
        check_mopac_was_success(output_str)

        logger.debug(f"Time taken: {time.time() - t0}")

        energy = _extract_energy(output_str)
        if energy is None:
            logger.info(output_str)
            raise MOPACError(f"Could not find energy in output string: {output_str}")

        if solvent == "cosmo2":
            E_nb, _ = calculate_nonpolar_term(out_path.read_text(), method="PM6")
            energy += E_nb
        return energy
