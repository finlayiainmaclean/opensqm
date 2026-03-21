# ruff: noqa: D100, PLR2004
import re

import numpy as np
import pandas as pd

from opensqm.mopac.exceptions import MOPACError


def _extract_energy(out_str: str) -> float | None:
    pattern = r"FINAL HEAT OF FORMATION\s*=\s*(-?\d+\.\d+)\s*KCAL/MOL"
    match = re.search(pattern, out_str)
    if match:
        energy = float(match.group(1))
    else:
        energy = None
    return energy


def calculate_nonpolar_term(output_file: str, /, *, method: str) -> tuple[float, float]:
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


def _extract_cosmo_area(output_str: str) -> float:
    """Extract COSMO area from MOPAC output file."""
    # Look for the COSMO AREA line
    match = re.search(r"COSMO AREA\s*=\s*([\d.]+)\s*SQUARE ANGSTROMS", output_str)

    if match:
        cosmo_area = float(match.group(1))
        return cosmo_area
    else:
        raise ValueError("COSMO AREA not found in output file")


def _extract_formal_charges_from_mopac_str(output_str: str) -> pd.DataFrame:
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


def _parse_bonds(text: str) -> set[tuple[int, int]]:
    """Parse atom connectivity text and return a set of bond tuples (i, j) with i < j."""
    if "TOPOGRAPHY OF SYSTEM" not in text and "Lewis Structure" not in text:
        raise MOPACError("Connectivity not found")

    text = text.split("TOPOGRAPHY OF SYSTEM", 1)[1].split("Lewis Structure", 1)[0]

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


def _extract_coords(text: str) -> pd.DataFrame:
    """Parse the final Cartesian table from MOPAC output.

    Optimization emits at least two CARTESIAN COORDINATES blocks (input + optimised).
    Slicing at the method citation line after the *last* block yields the optimised
    geometry.  Citations look like ``General Reference for PM6:`` or
    ``General Reference for PM6-D3H4X:``; an exact ``PM6:`` match misses D3H4X and
    the parser can swallow a second coordinate table (duplicate atom indices), giving
    2x atom rows.
    """
    start_marker = "CARTESIAN COORDINATES"
    # PM6, PM6-D3H4X, PM7, …
    end_at_citation = re.compile(
        r"^\s*General Reference for PM\d+(?:-[^\s:]+)?\s*:",
        flags=re.MULTILINE,
    )

    last_start = text.rfind(start_marker)
    if last_start == -1:
        raise MOPACError("Could not find coords to parse")

    section = text[last_start:]
    end_m = end_at_citation.search(section)
    if end_m is not None:
        section = section[: end_m.start()]
    else:
        for tail in (
            "\n          ATOMIC ORBITAL ELECTRON",
            "\n\n          ATOMIC ORBITAL ELECTRON",
            "\n == MOPAC DONE ==",
        ):
            i = section.find(tail)
            if i != -1:
                section = section[:i]
                break

    # MOZYME outputs often omit a PM citation immediately after the coordinate table.
    # The following ``NET ATOMIC CHARGES`` rows match the Cartesian regex (index,
    # element, three floats) and were being parsed as extra atom rows; the
    # duplicate-``1`` restart logic then kept only the charge block as “geometry”.
    _coord_block_end = re.compile(
        r"^\s*Empirical Formula\s*:",
        flags=re.MULTILINE,
    )
    _m = _coord_block_end.search(section)
    if _m is not None:
        section = section[: _m.start()]
    else:
        for tail in (
            "NET ATOMIC CHARGES AND DIPOLE",
            "NET ATOMIC CHARGES",
        ):
            i = section.find(tail)
            if i != -1:
                section = section[:i]
                break

    pattern = r"\s+(\d+)\s+([A-Z][a-z]?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    matches = re.findall(pattern, section)
    if not matches:
        raise MOPACError("Could not find coords to parse")

    df = pd.DataFrame(
        matches,
        columns=pd.Index(["atom_number", "symbol", "x", "y", "z"]),
    )
    df[["x", "y", "z"]] = df[["x", "y", "z"]].astype(float)
    # MOZYME / long outputs may still stack two numbered lists before the tail marker.
    atom_no = df["atom_number"].astype(int)
    restart = atom_no.index[atom_no == 1].tolist()
    if len(restart) > 1:
        df = df.iloc[restart[-1] :].reset_index(drop=True)

    df["atom_number"] = df["atom_number"].astype(int)
    df = df.sort_values("atom_number", kind="mergesort").reset_index(drop=True)
    nums = df["atom_number"].to_numpy()
    if nums.size and (nums[0] != 1 or not np.all(nums[1:] == nums[:-1] + 1)):
        raise MOPACError(
            f"coordinate table atom numbers are not 1..N consecutive (got {nums[:6]}...)"
        )
    return df
