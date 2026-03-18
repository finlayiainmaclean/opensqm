# ruff: noqa: D100, D101, D103, D205, D401, PLW2901, PLR2004, E501, PLC0415
import copy
import itertools
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Final, Literal

import mdtraj as md
import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from opensqm.rdkit_utils import set_coordinates
from opensqm.utils import run_command

OptMode = Literal["ligand", "pocket", "hydrogens"]


class MOPACError(Exception):
    pass


# Default van der Waals radii from MOZYME (in Angstroms)
MOPAC_VDW_RADII: Final[dict[str, float]] = {
    "H": 0.37,
    "He": 0.32,
    "Li": 1.34,
    "Be": 0.90,
    "B": 0.82,
    "C": 0.77,
    "N": 0.75,
    "O": 0.73,
    "F": 0.71,
    "Ne": 0.69,
    "Na": 1.54,
    "Mg": 1.30,
    "Al": 1.18,
    "Si": 1.11,
    "P": 1.06,
    "S": 1.02,
    "Cl": 0.99,
    "Ar": 0.97,
    "K": 1.96,
    "Ca": 1.74,
    "Sc": 1.44,
    "Ti": 1.36,
    "V": 1.25,
    "Cr": 1.27,
    "Mn": 1.39,
    "Fe": 1.25,
    "Co": 1.26,
    "Ni": 1.21,
    "Cu": 1.38,
    "Zn": 1.31,
    "Ga": 1.26,
    "Ge": 1.22,
    "As": 1.19,
    "Se": 1.16,
    "Br": 1.14,
    "Kr": 1.10,
    "Rb": 2.11,
    "Sr": 1.92,
    "Y": 1.62,
    "Zr": 1.48,
    "Nb": 1.37,
    "Mo": 1.45,
    "Tc": 1.56,
    "Ru": 1.26,
    "Rh": 1.35,
    "Pd": 1.31,
    "Ag": 1.53,
    "Cd": 1.48,
    "In": 1.44,
    "Sn": 1.41,
    "Sb": 1.38,
    "Te": 1.35,
    "I": 1.33,
    "Xe": 1.30,
    "Cs": 2.25,
    "Ba": 1.98,
    "La": 1.69,
    "Hf": 1.50,
    "Ta": 1.38,
    "W": 1.46,
    "Re": 1.59,
    "Os": 1.28,
    "Ir": 1.37,
    "Pt": 1.28,
    "Au": 1.44,
    "Hg": 1.49,
    "Tl": 1.48,
    "Pb": 1.47,
    "Bi": 1.46,
}


def fix_nitro_groups(mol: Chem.Mol) -> Chem.Mol:
    """
    Finds nitro groups in a molecule and adds single bonds between the two oxygens.

    See: https://github.com/openmopac/mopac/blob/main/src/chemistry/lewis.F90

    Args:
        mol: RDKit Mol object

    Returns
    -------
        RDKit Mol object with bonds added between nitro group oxygens
    """
    # Make a copy to avoid modifying the original molecule
    mol = Chem.Mol(mol)

    # Define SMARTS pattern for nitro group: [N+](=O)[O-]
    # This matches N with +1, double bonded to O, single bonded to O with -1
    nitro_pattern = Chem.MolFromSmarts("[N;D3](~[O;D1])(~[O;D1])")

    if nitro_pattern is None:
        return mol

    matches = mol.GetSubstructMatches(nitro_pattern)
    if not matches:
        return mol

    editable_mol = Chem.EditableMol(mol)
    bonds_added = set()

    for match in matches:
        n_idx, o1_idx, o2_idx = match

        if mol.GetAtomWithIdx(o1_idx).GetDegree() != 1:
            continue
        if mol.GetAtomWithIdx(o2_idx).GetDegree() != 1:
            continue

        bond_key = tuple(sorted([o1_idx, o2_idx]))
        if bond_key not in bonds_added:
            if mol.GetBondBetweenAtoms(o1_idx, o2_idx) is None:
                editable_mol.AddBond(o1_idx, o2_idx, Chem.BondType.SINGLE)
                bonds_added.add(bond_key)

    # Build modified mol from editable version
    modified_mol = editable_mol.GetMol()

    # Now zero charges on atoms in modified mol
    for match in matches:
        n_idx, o1_idx, o2_idx = match
        for idx in (n_idx, o1_idx, o2_idx):
            modified_mol.GetAtomWithIdx(idx).SetFormalCharge(0)

    return modified_mol


def _extract_energy(out_str):
    pattern = r"FINAL HEAT OF FORMATION\s*=\s*(-?\d+\.\d+)\s*KCAL/MOL"
    match = re.search(pattern, out_str)
    if match:
        energy = float(match.group(1))
    else:
        energy = None
    return energy


def calculate_nonpolar_term(output_file, /, *, method):
    """Calculate nonpolar term using COSMO area from MOPAC output."""
    # Surface tension parameter based on method
    if "PM6" in method:
        xi = 0.046
    elif "PM7" in method:
        xi = 0.042
    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract COSMO area
    cosmo_area = _extract_cosmo_area(output_file)

    # Calculate nonpolar contribution
    E_nb = cosmo_area * xi

    return E_nb, cosmo_area


def _extract_cosmo_area(output_str):
    """Extract COSMO area from MOPAC output file."""
    # Look for the COSMO AREA line
    match = re.search(r"COSMO AREA\s*=\s*([\d.]+)\s*SQUARE ANGSTROMS", output_str)

    if match:
        cosmo_area = float(match.group(1))
        return cosmo_area
    else:
        raise ValueError("COSMO AREA not found in output file")


def check_mopac_was_success(output_str):
    if "IMAGINARY FREQUENCIES" in output_str:
        raise MOPACError(f"IMAGINARY FREQUENCIES: {output_str}")
    if "EXCESS NUMBER OF OPTIMIZATION CYCLES" in output_str:
        raise MOPACError(f"EXCESS NUMBER OF OPTIMIZATION CYCLES: {output_str}")
    if "NOT ENOUGH TIME FOR ANOTHER CYCLE" in output_str:
        raise MOPACError(f"NOT ENOUGH TIME FOR ANOTHER CYCLE: {output_str}")
    success_keys = ["JOB ENDED NORMALLY", "MOPAC DONE"]
    correct_keys = all(key in output_str for key in success_keys)
    if correct_keys:
        return
    elif "A hydrogen atom is badly positioned" in output_str:
        raise MOPACError(f"Bad hydrogen: {output_str}")
    else:
        raise MOPACError(f"Unknown error: {output_str}")


def _extract_formal_charges_from_mopac_str(output_str):
    # regex: Ion, Atom No, Type, Charge
    pattern = re.compile(r"^\s*(\d+)\s+(\d+)\s+([A-Za-z]+)\s+([+-]?\d+)", re.MULTILINE)

    rows = []
    for match in pattern.finditer(output_str):
        ion, atom_no, atype, charge = match.groups()
        rows.append(
            {"ion": int(ion), "atom_ix": int(atom_no), "type": atype, "charge": int(charge)}
        )

    df = pd.DataFrame(rows)
    return df


def get_mopac_bonds(mol: Chem.Mol):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mol_mop_path = tmpdir / "mol.mop"
        setpi_path = tmpdir / "setpi.txt"
        mopac_path = tmpdir / "run.mopac"
        out_path = tmpdir / "run    ac.out"
        charge = Chem.GetFormalCharge(mol)

        mopac_str = f'LEWIS LET CHARGE={charge} GEO_DAT="{mol_mop_path!s}"'
        pi_bonds = get_mopac_pi_bonds(mol)
        write_setpi(pi_bonds, setpi_path)
        rdkit_to_mopac(mol, mol_mop_path)
        if len(pi_bonds) > 0:
            write_setpi(pi_bonds, setpi_path)
            mopac_str = f'{mopac_str} SETPI="{setpi_path}"'

        mopac_path.write_text(mopac_str)
        mopac_cmd = f"cd {tmpdir!s} && mopac {mopac_path!s}"
        run_command(mopac_cmd)
        output_str = out_path.read_text()
        check_mopac_was_success(output_str)

        bonds = _parse_bonds(output_str)

    return bonds


def _parse_bonds(text):
    """Parse atom connectivity text and return a set of bond tuples (i, j) with i < j."""
    if "TOPOGRAPHY OF SYSTEM" not in text and "Lewis Structure" not in text:
        raise MOPACError("Connectivity not found")

    text = text.split("TOPOGRAPHY OF SYSTEM")[1].split("Lewis Structure")[0]

    bonds = set()

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue  # skip lines without connectivity info

        if not parts[0].isdigit():
            continue  # skip non-atom-number lines

        atom_no = int(parts[0])
        connected_atoms = [int(a) for a in parts[2:] if a.isdigit()]

        for connected_no in connected_atoms:
            bond = tuple(sorted((atom_no, connected_no)))
            bonds.add(bond)

    return bonds


def get_atom_label(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    pdb_info = atom.GetPDBResidueInfo()
    if pdb_info is None:
        raise ValueError(f"Atom {atom_idx} has no PDB residue info.")

    resname = pdb_info.GetResidueName().strip()  # e.g., "PRO"
    resnum = pdb_info.GetResidueNumber()  # e.g., 7
    atomname = pdb_info.GetName().strip()  # e.g., "CA"
    return f"{resname}-{resnum}-{atomname}"


def get_cvb_str(mol: Chem.Mol):

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
        mol_mop_path = tmpdir / "mol.mop"
        setpi_path = tmpdir / "setpi.txt"
        mopac_path = tmpdir / "run.mopac"
        out_path = tmpdir / "run    ac.out"
        charge = Chem.GetFormalCharge(mol)

        base_mopac_str = f'LEWIS LET METAL CHARGE={charge} GEO_DAT="{mol_mop_path!s}"'
        pi_bonds = get_mopac_pi_bonds(mol)
        write_setpi(pi_bonds, setpi_path)
        rdkit_to_mopac(mol, mol_mop_path)
        if len(pi_bonds) > 0:
            write_setpi(pi_bonds, setpi_path)
            base_mopac_str = f'{base_mopac_str} SETPI="{setpi_path}"'

        for bond_str in bonds_str:
            _bond_str = f"CVB({bond_str})"

            mopac_str = f"{base_mopac_str} {_bond_str}"
            mopac_path.write_text(mopac_str)
            mopac_cmd = f"cd {tmpdir!s} && mopac {mopac_path!s}"
            run_command(mopac_cmd)
            output_str = out_path.read_text()

            if "-" in _bond_str:  # Deleting a bond
                if "did not exist" in output_str:
                    logger.info(f"Bond to delete doesn't exist: {bond_str}")

                    # import pdb; pdb.set_trace()
                    continue
                else:
                    # "have been deleted" in output_str:
                    logger.info(f"Deleting bond: {bond_str}")
                    valid_bonds_str.append(bond_str)
                # else:
                #     logger.info("Other error")
                #     import pdb; pdb.set_trace()
                #     raise ValueError("Did not detect either keyword is bond deletion worked.")

            elif "already exists" in output_str:
                logger.info(f"Bond to add already exists: {bond_str}")
                # import pdb; pdb.set_trace()
                continue
            elif "This is unrealistic" in output_str:
                logger.info(f"Bond too add is too long: {bond_str}")
                # import pdb; pdb.set_trace()
                continue
            else:
                logger.info(f"Adding bond: {bond_str}")
                valid_bonds_str.append(bond_str)

    cvb_str = ""
    if len(valid_bonds_str) > 0:
        bonds_str = ";".join(valid_bonds_str)
        cvb_str = f"CVB({bonds_str})"

    # import pdb; pdb.set_trace()

    return cvb_str


def get_opt_mask(complex: Chem.Mol, mode: OptMode = "ligand"):

    opt_mask = np.zeros(complex.GetNumAtoms(), dtype=bool)

    match mode:
        case "ligand":
            opt_mask |= np.array(
                [a.GetPDBResidueInfo().GetResidueName() == "LIG" for a in complex.GetAtoms()]
            )

        case "pocket":
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_com = Path(tmpdir) / "com.pdb"
                Chem.MolToPDBFile(complex, str(tmp_com))
                traj = md.load(str(tmp_com))
                top = traj.topology
                lig_atoms = top.select("resname LIG")
                sidechain_atoms = top.select("sidechain")
                pairs = md.compute_neighbors(
                    traj, cutoff=0.4, query_indices=lig_atoms, haystack_indices=sidechain_atoms
                )
                close_atoms = set(pairs[0].tolist())
                close_atoms = close_atoms | set(lig_atoms)
                opt_mask[list(close_atoms)] = True
        case "hydrogens":
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_com = Path(tmpdir) / "com.pdb"
                Chem.MolToPDBFile(complex, str(tmp_com))
                traj = md.load(str(tmp_com))
                top = traj.topology
                lig_atoms = top.select("resname LIG")
                lig_h_atoms = top.select("resname LIG and element H")
                h_atoms = top.select("element H")
                pairs = md.compute_neighbors(
                    traj, cutoff=0.4, query_indices=lig_atoms, haystack_indices=h_atoms
                )
                close_atoms = set(pairs[0].tolist())
                close_atoms = close_atoms - set(lig_atoms)
                close_atoms = close_atoms | set(lig_h_atoms)
                opt_mask[list(close_atoms)] = True

    return opt_mask.astype(int)


def annotate_mopac_pi_bonds(mol: Chem.Mol, /, *, bonds: list[tuple[int, int]]):
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


def get_correct_ligand(ligand):
    """Find the MOZYME ligand configuration that best approximates the SCF energy *without* the MOZYME approximation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mol_mop_path = tmpdir / "mol.mop"
        setpi_path = tmpdir / "setpi.txt"
        mopac_path = tmpdir / "run.mopac"
        out_path = tmpdir / "run    ac.out"

        rdkit_formal_charge = Chem.GetFormalCharge(ligand)
        rdkit_to_mopac(ligand, mol_mop_path)

        cvb_str = get_cvb_str(ligand)

        mopac_str = f'PM6 1SCF CHARGE={rdkit_formal_charge} GEO_DAT="{mol_mop_path!s}" {cvb_str}'

        mopac_path.write_text(mopac_str)
        mopac_cmd = f"cd {tmpdir!s} && mopac {mopac_path!s}"
        run_command(mopac_cmd)
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
            rdkit_to_mopac(_ligand, mol_mop_path)

            mopac_str = (
                f'PM6 MOZYME 1SCF CHARGE={rdkit_formal_charge} GEO_DAT="{mol_mop_path!s}" {cvb_str}'
            )
            if len(_pi_bonds) > 0:
                mopac_str = f'{mopac_str} SETPI="{setpi_path!s}"'
            mopac_path.write_text(mopac_str)
            mopac_cmd = f"cd {tmpdir!s} && mopac {mopac_path!s}"
            run_command(mopac_cmd)
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


def annotate_mopac_formal_charges(mol, formal_charges):
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetIntProp("mopac_formal_charge", formal_charges.get(i, 0))


def rdkit_to_mopac(mol, out_mopac_path, opt_mask=None, conf_id=0):
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


def get_rdkit_formal_charges(mol):
    charges = {}
    for i, atom in enumerate(mol.GetAtoms()):
        formal_charge = atom.GetFormalCharge()
        if formal_charge != 0:
            charges[i] = formal_charge
    return charges


def all_combinations(items):
    result = [{}]
    for length in range(1, len(items) + 1):
        result.extend(itertools.combinations(items, length))
    result.sort(key=len)
    return result


def get_rdkit_pi_bonds(mol):
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


def write_setpi(pi_bonds, setpi_path):
    pi_bonds_formatted = [f"{a1 + 1} {a2 + 1}" for a1, a2 in pi_bonds]
    setpi_path.write_text("\n".join(pi_bonds_formatted))


def _extract_coords(text):
    # Extract section between the markers
    start_marker = "CARTESIAN COORDINATES"
    end_marker = "General Reference for PM6:"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        section = text[start_idx:end_idx]

        # Find all lines with atom data (number, symbol, x, y, z)
        pattern = r"\s+(\d+)\s+([A-Z][a-z]?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
        matches = re.findall(pattern, section)

        # Create DataFrame
        df = pd.DataFrame(
            matches,
            columns=pd.Index(["atom_number", "symbol", "x", "y", "z"]),
        )

        # Convert coordinates to float
        df[["x", "y", "z"]] = df[["x", "y", "z"]].astype(float)

    else:
        raise MOPACError("Could not find coords ti parse")
    return df


def run_opt_from_rdmol(rdmol, /, *, mopac_keywords: list[str], charge=0, opt_mask=None) -> Chem.Mol:

    rdmol = Chem.Mol(rdmol)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mop_path = tmpdir / "mol.mop"
        setpi_path = tmpdir / "setpi.txt"
        mopac_path = tmpdir / "run.mopac"
        tmpdir / "run    ac.pdb"
        out_path = tmpdir / "run    ac.out"

        pi_bonds = get_mopac_pi_bonds(rdmol)
        write_setpi(pi_bonds, setpi_path)
        rdkit_to_mopac(rdmol, mop_path, opt_mask=opt_mask)
        if len(pi_bonds) > 0:
            write_setpi(pi_bonds, setpi_path)
            mopac_keywords.append(f'SETPI="{setpi_path}"')
        cvb_str = get_cvb_str(rdmol)

        mopac_keywords.append(f"CHARGE={charge}")
        mopac_keywords.append(f'GEO_DAT="{mop_path!s}"')
        mopac_keywords.append(cvb_str)

        mopac_str = " ".join(mopac_keywords)

        logger.debug(f"Optimisation complex calculation:\n{mopac_str}")
        t0 = time.time()
        mopac_path.write_text(mopac_str)
        mopac_cmd = f"mopac {mopac_path!s}"

        run_command(mopac_cmd, cwd=tmpdir)
        check_mopac_was_success(out_path.read_text())

        logger.debug(f"Time taken: {time.time() - t0}")

        df = _extract_coords(out_path.read_text())

        coords = df[["x", "y", "z"]].values

        for a, symbol in zip(rdmol.GetAtoms(), df.symbol, strict=False):
            assert a.GetSymbol() == symbol

        rdmol = set_coordinates(rdmol, coords=coords)

        return rdmol


def run_singlepoint_from_rdmol(
    rdmol,
    /,
    *,
    use_mozyme: bool = True,
    solvent: Literal["cosmo", "cosmo2"] | None = "cosmo2",
    charge: int = 0,
):

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

    energy_pattern = r"FINAL HEAT OF FORMATION\s*=\s*(-?\d+\.\d+)\s*KCAL/MOL"

    mopac_keywords = copy.deepcopy(mopac_keywords)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mop_path = tmpdir / "mol.mop"
        setpi_path = tmpdir / "setpi.txt"
        mopac_path = tmpdir / "run.mopac"
        out_path = tmpdir / "run    ac.out"

        rdkit_to_mopac(rdmol, mop_path)

        cvb_str = get_cvb_str(rdmol)

        mopac_keywords.append(f"CHARGE={charge}")
        mopac_keywords.append(f'GEO_DAT="{mop_path}"')
        mopac_keywords.append(cvb_str)

        pi_bonds = get_mopac_pi_bonds(rdmol)
        if len(pi_bonds) > 0:
            write_setpi(pi_bonds, setpi_path)
            mopac_keywords.append(f'SETPI="{setpi_path}"')

        mopac_str = " ".join(mopac_keywords)

        mopac_path.write_text(mopac_str)
        logger.debug(f"Singlepoint ligand calculation:\n{mopac_str}")
        t0 = time.time()
        mopac_cmd = f"mopac {mopac_path!s}"

        run_command(mopac_cmd, cwd=tmpdir)
        output_str = out_path.read_text()
        check_mopac_was_success(output_str)

        logger.debug(f"Time taken: {time.time() - t0}")

        match = re.search(energy_pattern, output_str)
        if match:
            energy = float(match.group(1))
        else:
            logger.info(output_str)
            raise MOPACError(f"Could not find enerrgy in output string: {output_str}")

        if solvent == "cosmo2":
            E_nb, _ = calculate_nonpolar_term(out_path.read_text(), method="PM6")
            energy += E_nb
        return energy


def get_mopac_formal_charges(rdmol):

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mol_mop_path = tmpdir / "mol.mop"
        mopac_path = tmpdir / "run.mopac"
        out_path = tmpdir / "run    ac.out"
        setpi_path = tmpdir / "setpi.txt"

        rdkit_to_mopac(rdmol, mol_mop_path)

        cvb_str = get_cvb_str(rdmol)

        mopac_str = f'CHARGES LET GEO_DAT="{mol_mop_path!s}" {cvb_str}'

        pi_bonds = get_mopac_pi_bonds(rdmol)
        if len(pi_bonds) > 0:
            write_setpi(pi_bonds, setpi_path)
            mopac_str = f'{mopac_str} SETPI="{setpi_path}"'

        mopac_path.write_text(mopac_str)
        mopac_cmd = f"cd {tmpdir!s} && mopac {mopac_path!s}"
        run_command(mopac_cmd)
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


def run_interaction_energy(*, ligand: Chem.Mol, protein: Chem.Mol) -> dict[str, float]:

    # Get ligand with annoted formal charges and pi bonds for mopac
    ligand, _dE = get_correct_ligand(ligand)

    rdkit_ligand_charge = Chem.GetFormalCharge(ligand)
    n_ligand_atoms = ligand.GetNumAtoms()

    # Get ligand charges from both rdkit and mopac and check
    # rdkit_ligand_formal_charges = get_rdkit_formal_charges(ligand)
    mopac_ligand_formal_charge, mopac_ligand_formal_charges = get_mopac_formal_charges(ligand)

    if mopac_ligand_formal_charge != rdkit_ligand_charge:
        print("RDKit and MOPAC disagree on ligand charge")
        Chem.MolToMolFile(ligand, "/tmp/ligand.mol")
        import pdb

        pdb.set_trace()

    # RDKit does not parse salt bridges well, so use MOPAC to find atom formal charges
    mopac_protein_charge, mopac_protein_formal_charges = get_mopac_formal_charges(protein)
    # Annotate the protein with these charges (strictly unneccesary as this is a round-trip
    annotate_mopac_formal_charges(protein, mopac_protein_formal_charges)
    complex_charge = rdkit_ligand_charge + mopac_protein_charge

    # Build the complex, making sure the ligand is first, as mopac will use the 1-indexed pi bonds
    complex = Chem.CombineMols(ligand, protein)

    # Annotate the complex with the rdkit-derived formal charges of the ligand
    # and the mopac-derived charges of the protein.
    # We need to shift the atom indices of the protein by N_atoms of the ligand
    # TODO(fin): Split cofactors to be treated like the ligand
    complex_formal_charges = mopac_ligand_formal_charges
    for protein_atom_ix, protein_atom_charge in mopac_protein_formal_charges.items():
        complex_atom_ix = protein_atom_ix + n_ligand_atoms
        complex_formal_charges[complex_atom_ix] = protein_atom_charge
    annotate_mopac_formal_charges(complex, complex_formal_charges)

    E_ligand = run_singlepoint_from_rdmol(
        ligand, use_mozyme=True, solvent="cosmo2", charge=rdkit_ligand_charge
    )
    E_protein = run_singlepoint_from_rdmol(
        protein, use_mozyme=True, solvent="cosmo2", charge=mopac_protein_charge
    )
    E_complex = run_singlepoint_from_rdmol(
        complex, use_mozyme=True, solvent="cosmo2", charge=complex_charge
    )
    dE_int = E_complex - E_protein - E_ligand

    scores = {
        "dE_int": dE_int,
        "E_complex": E_complex,
        "E_protein": E_protein,
        "E_ligand": E_ligand,
    }

    return scores


def optimise_complex(
    *,
    ligand: Chem.Mol,
    protein: Chem.Mol,
    mode: OptMode,
    gnorm: float = 1.0,
    use_rapid: bool = False,
    num_epochs: int = 30,
):

    mopac_keywords = [
        "PM6-D3H4X",
        "MOZYME",
        f"LET({num_epochs})",
        "LBFGS",
        "RHF",
        "METAL",
        "NOMM",
        "EPS=78.5",
        f"GNORM={gnorm}",
    ]

    if use_rapid:
        mopac_keywords.append("RAPID")

    # Get ligand with annoted formal charges and pi bonds for mopac
    ligand, _dE = get_correct_ligand(ligand)

    rdkit_ligand_charge = Chem.GetFormalCharge(ligand)
    n_ligand_atoms = ligand.GetNumAtoms()

    # Get ligand charges from both rdkit and mopac and check
    ligand_formal_charges = get_rdkit_formal_charges(ligand)
    _, mopac_ligand_formal_charges = get_mopac_formal_charges(ligand)
    assert mopac_ligand_formal_charges == ligand_formal_charges, (
        "RDKit and MOPAC disagree on ligand charges"
    )

    # RDKit does not parse salt bridges well, so use MOPAC to find atom formal charges
    mopac_protein_charge, mopac_protein_formal_charges = get_mopac_formal_charges(protein)
    # Annotate the protein with these charges (strictly unneccesary as this is a round-trip
    annotate_mopac_formal_charges(protein, mopac_protein_formal_charges)
    complex_charge = rdkit_ligand_charge + mopac_protein_charge

    # Build the complex, making sure the ligand is first, as mopac will use the 1-indexed pi bonds
    complex = Chem.CombineMols(ligand, protein)

    AllChem.MMFFOptimizeMolecule(complex)  # type: ignore[unresolved-attribute]

    # Annotate the complex with the rdkit-derived formal charges of the ligand
    # and the mopac-derived charges of the protein.
    # We need to shift the atom indices of the protein by N_atoms of the ligand
    # TODO(fin): Split cofactors to be treated like the ligand
    complex_formal_charges = mopac_ligand_formal_charges
    for protein_atom_ix, protein_atom_charge in mopac_protein_formal_charges.items():
        complex_atom_ix = protein_atom_ix + n_ligand_atoms
        complex_formal_charges[complex_atom_ix] = protein_atom_charge
    annotate_mopac_formal_charges(complex, complex_formal_charges)

    opt_mask = get_opt_mask(complex, mode=mode)
    logger.info(f"{sum(opt_mask)} atoms to optimise")

    complex_optimised = run_opt_from_rdmol(
        complex, opt_mask=opt_mask, mopac_keywords=mopac_keywords, charge=complex_charge
    )

    # Extract ligand and protein conformations from the optimised complex
    ligand_idxs = [
        atom.GetIdx()
        for atom in complex_optimised.GetAtoms()
        if atom.GetPDBResidueInfo().GetResidueName() == "LIG"
    ]

    protein_idxs = [
        atom.GetIdx()
        for atom in complex_optimised.GetAtoms()
        if atom.GetPDBResidueInfo().GetResidueName() != "LIG"
    ]

    rw = Chem.RWMol(complex_optimised)
    # Remove all ligand atoms to get protein
    for idx in sorted(ligand_idxs, reverse=True):
        rw.RemoveAtom(idx)
    protein_opt = rw.GetMol()

    rw = Chem.RWMol(complex_optimised)
    # Remove all protein atoms to get ligand
    for idx in sorted(protein_idxs, reverse=True):
        rw.RemoveAtom(idx)
    ligand_opt = rw.GetMol()

    ligand_rmsd = rdMolAlign.CalcRMS(ligand, ligand_opt)
    protein_rmsd = rdMolAlign.CalcRMS(protein, protein_opt)

    logger.info(f"Ligand RMSD: {ligand_rmsd:.2f}")
    logger.info(f"Protein RMSD: {protein_rmsd:.2f}")

    return ligand_opt, protein_opt
