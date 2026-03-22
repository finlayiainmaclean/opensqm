# ruff: noqa: D100, D103, D205, PLR2004, E501
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem


def annotate_mopac_pi_bonds(mol: Chem.Mol, /, *, bonds: list[tuple[int, int]]) -> None:
    """Annotate bonds in `mol` with (src, tgt) tuples as a property."""
    prop_name = "mopac_pi_bonds"
    for src, tgt in bonds:
        bond = mol.GetBondBetweenAtoms(src, tgt)
        if bond is None:
            raise ValueError(f"No bond between atoms {src} and {tgt}")
        bond.SetProp(prop_name, json.dumps((src, tgt)))


def get_mopac_pi_bonds(mol: Chem.Mol) -> list[tuple[int, int]]:
    """Extract (src, tgt) tuples stored as bond properties."""
    prop_name = "mopac_pi_bonds"
    pairs = []
    for bond in mol.GetBonds():
        if bond.HasProp(prop_name):
            pairs.append(tuple(json.loads(bond.GetProp(prop_name))))
    return pairs


def annotate_mopac_formal_charges(mol: Chem.Mol, formal_charges: dict[int, int]) -> None:
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetIntProp("mopac_formal_charge", formal_charges.get(i, 0))


def get_rdkit_formal_charges(mol: Chem.Mol) -> dict[int, int]:
    charges = {}
    for i, atom in enumerate(mol.GetAtoms()):
        formal_charge = atom.GetFormalCharge()
        if formal_charge != 0:
            charges[i] = formal_charge
    return charges


def all_combinations(items: list[tuple[int, int]]) -> list[Any]:
    result = [{}]
    for length in range(1, len(items) + 1):
        result.extend(itertools.combinations(items, length))
    result.sort(key=len)
    return result


def get_rdkit_pi_bonds(mol: Chem.Mol) -> list[tuple[int, int]]:
    """
    Return list of (1-indexed begin_atom, end_atom) strings
    for double bonds where both atoms.
    """
    pi_bonds = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if bond.GetBondTypeAsDouble() == 2.0:
            pi_bonds.append((a1.GetIdx(), a2.GetIdx()))
    return pi_bonds


def write_setpi(pi_bonds: list[tuple[int, int]], setpi_path: Path) -> None:
    pi_bonds_formatted = [f"{a1 + 1} {a2 + 1}" for a1, a2 in pi_bonds]
    setpi_path.write_text("\n".join(pi_bonds_formatted))


def _pi_bonds_prepare_setpi_file(mol: Chem.Mol, setpi_path: Path) -> list[tuple[int, int]]:
    """Write `setpi.txt` from annotated π bonds on ``mol``; call before ``rdkit_to_mopac``."""
    pi = get_mopac_pi_bonds(mol)
    write_setpi(pi, setpi_path)
    return pi


def _finalize_setpi_after_geometry(
    pi_bonds: list[tuple[int, int]],
    setpi_path: Path,
    *,
    mopac_keywords: list[str] | None = None,
) -> str:
    """Refresh ``setpi.txt`` after geometry exists. Optionally append SETPI to ``mopac_keywords``."""
    if len(pi_bonds) == 0:
        return ""
    write_setpi(pi_bonds, setpi_path)
    token = f'SETPI="{setpi_path!s}"'
    if mopac_keywords is not None:
        mopac_keywords.append(token)
        return ""
    return f" {token}"


def rdkit_to_mopac(
    mol: Chem.Mol,
    out_mopac_path: Path,
    opt_mask: np.ndarray | None = None,
    conf_id: int = 0,
) -> None:
    # Get the conformer
    conf = mol.GetConformer(conf_id)

    mopac_lines = [
        "PM6",
        "",
        "",
    ]

    if opt_mask is None:
        opt_mask = np.ones(mol.GetNumAtoms())

    opt_mask = opt_mask.astype(int)
    assert len(opt_mask) == mol.GetNumAtoms()

    # Add atomic coordinates
    for i, atom in enumerate(mol.GetAtoms()):
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        symbol = atom.GetSymbol()
        charge = (
            atom.GetIntProp("mopac_formal_charge") if atom.HasProp("mopac_formal_charge") else 0
        )
        if charge > 0:
            symbol += "(+)"
        elif charge < 0:
            symbol += "(-)"

        # MOPAC format: Symbol x opt y opt z opt
        # opt can be 1 (optimize) or 0 (freeze)
        optimise_atom = opt_mask[i]
        coord_line = f"{symbol:2s} {pos.x:12.6f} {optimise_atom} {pos.y:12.6f} {optimise_atom} {pos.z:12.6f} {optimise_atom}"
        mopac_lines.append(coord_line)

    out_mopac_path.write_text("\n".join(mopac_lines))
