# ruff: noqa: D100, PLW2901, PLR2004, E501
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger
from rdkit import Chem

from opensqm.mopac.charges import get_mopac_formal_charges
from opensqm.mopac.geometry import (
    _finalize_setpi_after_geometry,
    _pi_bonds_prepare_setpi_file,
    all_combinations,
    annotate_mopac_formal_charges,
    annotate_mopac_pi_bonds,
    get_rdkit_formal_charges,
    get_rdkit_pi_bonds,
    rdkit_to_mopac,
    write_setpi,
)
from opensqm.mopac.lewis import get_cvb_str
from opensqm.mopac.nitro import fix_nitro_groups
from opensqm.mopac.parse_output import _extract_energy
from opensqm.mopac.runner import _run_mopac_input_file, check_mopac_was_success


def get_correct_ligand(ligand: Chem.Mol) -> tuple[Chem.Mol, float]:
    """Find the MOZYME ligand configuration that best approximates the SCF energy *without* the MOZYME approximation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mol_mop_path = tmpdir / "mol.mop"
        setpi_path = tmpdir / "setpi.txt"
        mopac_path = tmpdir / "run.mopac"
        out_path = tmpdir / "run    ac.out"

        rdkit_formal_charge = Chem.GetFormalCharge(ligand)
        ligand_mop = fix_nitro_groups(ligand)
        pi_bonds_init = _pi_bonds_prepare_setpi_file(ligand_mop, setpi_path)
        rdkit_to_mopac(ligand_mop, mol_mop_path)

        cvb_str = get_cvb_str(ligand_mop)

        mopac_str = f'PM6 1SCF CHARGE={rdkit_formal_charge} GEO_DAT="{mol_mop_path!s}" {cvb_str}'
        mopac_str += _finalize_setpi_after_geometry(pi_bonds_init, setpi_path)

        mopac_path.write_text(mopac_str)
        _run_mopac_input_file(mopac_path, cwd=tmpdir)
        output_str = out_path.read_text()
        check_mopac_was_success(output_str)
        energy = _extract_energy(output_str)
        if energy is None:
            raise ValueError("Failed to extract energy from MOPAC output")

        pi_bonds = get_rdkit_pi_bonds(ligand)
        pi_bond_combos = all_combinations(pi_bonds)[:20]

        dEs = []
        successful_ligands = []

        for _pi_bonds in pi_bond_combos:
            _ligand = Chem.Mol(ligand)
            annotate_mopac_pi_bonds(_ligand, bonds=_pi_bonds)
            write_setpi(_pi_bonds, setpi_path)

            _formal_charges = get_rdkit_formal_charges(_ligand)
            annotate_mopac_formal_charges(_ligand, _formal_charges)
            _ligand_mop = fix_nitro_groups(_ligand)
            rdkit_to_mopac(_ligand_mop, mol_mop_path)

            mopac_str = (
                f'PM6 MOZYME 1SCF CHARGE={rdkit_formal_charge} GEO_DAT="{mol_mop_path!s}" {cvb_str}'
            )
            mopac_str += _finalize_setpi_after_geometry(_pi_bonds, setpi_path)
            mopac_path.write_text(mopac_str)
            _run_mopac_input_file(mopac_path, cwd=tmpdir)
            output_str = out_path.read_text()
            check_mopac_was_success(output_str)
            mozyme_energy = _extract_energy(output_str)

            mopac_ligand_formal_charge, _ = get_mopac_formal_charges(_ligand)

            if mopac_ligand_formal_charge != rdkit_formal_charge:
                logger.info("Formal charge mismatch")
                continue

            if mozyme_energy is not None:
                successful_ligands.append(_ligand)
                dE = abs(mozyme_energy - energy)
                dEs.append(dE)
                if dE < 10:
                    break
            else:
                logger.info("Failed to converge")

        if len(dEs) == 0:
            raise ValueError("Failed to find a single converged SCF for the input ligand")

        best_idx = np.nanargmin(dEs)
        best_dE = dEs[best_idx]
        best_ligand = successful_ligands[best_idx]

        return best_ligand, best_dE
