# ruff: noqa: D100, D103, PLW2901, E501
import tempfile
from pathlib import Path

from loguru import logger
from rdkit import Chem

from opensqm.mopac.geometry import (
    _finalize_setpi_after_geometry,
    _pi_bonds_prepare_setpi_file,
    rdkit_to_mopac,
)
from opensqm.mopac.nitro import fix_nitro_groups
from opensqm.mopac.parse_output import _parse_bonds
from opensqm.mopac.runner import _run_mopac_input_file, check_mopac_was_success


def _build_lewis_geo_dat_keywords(mol: Chem.Mol, tmpdir: Path, *, metal: bool) -> str:
    """Write `mol.mop` (+ optional `setpi.txt`) and return the LEWIS keyword line (GEO_DAT, optional SETPI)."""
    mol_mop_path = tmpdir / "mol.mop"
    setpi_path = tmpdir / "setpi.txt"
    charge = Chem.GetFormalCharge(mol)
    if metal:
        mopac_str = f'LEWIS LET METAL CHARGE={charge} GEO_DAT="{mol_mop_path!s}"'
    else:
        mopac_str = f'LEWIS LET CHARGE={charge} GEO_DAT="{mol_mop_path!s}"'
    pi_bonds = _pi_bonds_prepare_setpi_file(mol, setpi_path)
    rdkit_to_mopac(mol, mol_mop_path)
    return mopac_str + _finalize_setpi_after_geometry(pi_bonds, setpi_path)


def _prepare_mopac_lewis_bonds_job(mol: Chem.Mol, tmpdir: Path) -> tuple[Path, Path]:
    """Write LEWIS + GEO_DAT (+ optional SETPI) job files; return (control path, expected .out path)."""
    mopac_path = tmpdir / "run.mopac"
    out_path = tmpdir / "run    ac.out"
    mopac_path.write_text(_build_lewis_geo_dat_keywords(mol, tmpdir, metal=False))
    return mopac_path, out_path


def get_mopac_bonds(mol: Chem.Mol) -> set[tuple[int, int]]:
    mol = fix_nitro_groups(mol)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mopac_path, out_path = _prepare_mopac_lewis_bonds_job(mol, tmpdir)
        _run_mopac_input_file(mopac_path, cwd=tmpdir)
        output_str = out_path.read_text()
        check_mopac_was_success(output_str)
        return _parse_bonds(output_str)


def get_atom_label(mol: Chem.Mol, atom_idx: int) -> str:
    atom = mol.GetAtomWithIdx(atom_idx)
    pdb_info = atom.GetPDBResidueInfo()
    if pdb_info is None:
        raise ValueError(f"Atom {atom_idx} has no PDB residue info.")

    resname = pdb_info.GetResidueName().strip()  # e.g., "PRO"
    resnum = pdb_info.GetResidueNumber()  # e.g., 7
    atomname = pdb_info.GetName().strip()  # e.g., "CA"
    return f"{resname}-{resnum}-{atomname}"


def get_cvb_str(mol: Chem.Mol) -> str:
    mol = fix_nitro_groups(mol)

    mopac_bonds = get_mopac_bonds(mol)

    # Extract reference bonds from RDKit mol
    rdkit_bonds = {
        tuple(sorted([b.GetBeginAtomIdx() + 1, b.GetEndAtomIdx() + 1])) for b in mol.GetBonds()
    }

    extra_bonds = mopac_bonds - rdkit_bonds
    missing_bonds = rdkit_bonds - mopac_bonds

    for s, t in extra_bonds:
        logger.info(f"Extra bond: {get_atom_label(mol, s - 1)}//{get_atom_label(mol, t - 1)}")

    for s, t in missing_bonds:
        logger.info(f"Missing bond: {get_atom_label(mol, s - 1)}//{get_atom_label(mol, t - 1)}")

    missing_bonds_str = [f"{i}:{j}" for i, j in missing_bonds]
    extra_bonds_str = [f"{i}:-{j}" for i, j in extra_bonds]

    if len(extra_bonds) > 0:
        logger.info(f"Has extra bonds {','.join(extra_bonds_str)}")
    if len(missing_bonds) > 0:
        logger.info(f"Has missing bonds {','.join(missing_bonds_str)}")

    bonds_str = missing_bonds_str + extra_bonds_str

    valid_bonds_str = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mopac_path = tmpdir / "run.mopac"
        out_path = tmpdir / "run    ac.out"
        # Writes mol.mop / setpi.txt; string includes SETPI="…/setpi.txt" when π-bonds are annotated.
        base_mopac_str = _build_lewis_geo_dat_keywords(mol, tmpdir, metal=True)

        for bond_str in bonds_str:
            _bond_str = f"CVB({bond_str})"

            mopac_str = f"{base_mopac_str} {_bond_str}"
            mopac_path.write_text(mopac_str)
            _run_mopac_input_file(mopac_path, cwd=tmpdir)
            output_str = out_path.read_text()

            if "-" in _bond_str:  # Deleting a bond
                if "did not exist" in output_str:
                    logger.info(f"Bond to delete doesn't exist: {bond_str}")
                    continue
                else:
                    logger.info(f"Deleting bond: {bond_str}")
                    valid_bonds_str.append(bond_str)

            elif "already exists" in output_str:
                logger.info(f"Bond to add already exists: {bond_str}")
                continue
            elif "This is unrealistic" in output_str:
                logger.info(f"Bond too add is too long: {bond_str}")
                continue
            else:
                logger.info(f"Adding bond: {bond_str}")
                valid_bonds_str.append(bond_str)

    cvb_str = ""
    if len(valid_bonds_str) > 0:
        bonds_str = ";".join(valid_bonds_str)
        cvb_str = f"CVB({bonds_str})"

    return cvb_str
