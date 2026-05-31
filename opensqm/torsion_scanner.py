"""Module for detecting Type 2 atropisomers using torsion scans."""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
from openff.toolkit.topology import Molecule
from openmm import CustomTorsionForce, LangevinMiddleIntegrator, app, unit
from openmm.app import Simulation
from rdkit import Chem
from rdkit.Chem import AllChem

from opensqm.md.prepare import get_ligand_forcefield

logger = logging.getLogger(__name__)

# Base constants for Eyring
KB_T_OVER_H = 6.2e12  # k_B * T / h at 300K in s^-1
R_KCAL = 0.001987  # kcal / (mol * K)
T_K = 300.0  # K
EV_2_KCAL = 23.0605


def get_half_life(barrier_kcal: float, temp_k: float = T_K) -> float:
    """Calculate half life in seconds for a given barrier height in kcal/mol."""
    rate = KB_T_OVER_H * np.exp(-barrier_kcal / (R_KCAL * temp_k))
    return np.log(2) / rate


def get_barrier_kcal(half_life: float, temp_k: float = T_K) -> float:
    """Calculate barrier height in kcal/mol for a given half life in seconds."""
    rate = np.log(2) / half_life
    return -R_KCAL * temp_k * np.log(rate / KB_T_OVER_H)


def is_type_2_atropisomer(barrier_kcal: float, min_hl: float = 1e-9, max_hl: float = 2.5e6) -> bool:
    """
    Check if a barrier falls within the Type 2 atropisomer timeframe.

    min_hl: 1 ns (too slow for MD to capture)
    max_hl: ~1 mo (cross organically during assay but not indefinitely stable).
    """
    hl = get_half_life(barrier_kcal)
    return min_hl < hl < max_hl


def get_rotatable_terminal_bonds(rdmol: Chem.Mol) -> list[tuple[int, int]]:
    """
    Find rotatable bonds conceptually matching ring attachments.

    Uses SMARTs to identify acyclic single bonds between distinct heavy-atom cores.
    """
    # Looks for any single acyclic bond! You can prune this SMARTs further
    # to target exact C2-symmetric functional groups like rings: '[!R]-!@[c1ccccc1]'
    rotatable_smarts = Chem.MolFromSmarts("[!#1]~[!$(*#*)&!D1:1]-,=;!@[!$(*#*)&!D1:2]~[!#1]")
    matches = rdmol.GetSubstructMatches(rotatable_smarts)

    bonds = set()
    # Avoid duplicates
    for m in matches:
        _a, b, c, _d = m

        b, c = tuple(sorted((b, c)))
        # a,b,c,d = tuple(sorted(m))
        if (b, c) not in bonds:
            bonds.add((b, c))

    return list(bonds)


def torsion_barriers(
    angles: np.ndarray,
    energies: np.ndarray,
    energy_window: Optional[float] = 10.0,
) -> Dict[str, Any]:
    """
    Identify torsional minima and local barriers between adjacent minima.

    Parameters
    ----------
    angles : np.ndarray
        1D array of torsion angles (degrees). Does not need to be sorted.
    energies : np.ndarray
        1D array of relative energies (kcal/mol), same length as angles.
    smooth : bool, optional
        Apply Savitzky-Golay smoothing before extrema detection.
    window : int, optional
        Window length for smoothing (must be odd).
    poly : int, optional
        Polynomial order for smoothing.
    energy_window : float or None, optional
        Keep minima within this many kcal/mol of global minimum.
        Set to None to keep all minima.

    Returns
    -------
    dict with:
        minima : list of dicts
        barriers : list of dicts
    """
    angles = np.asarray(angles)
    energies = np.asarray(energies)

    # --- sort ---
    idx = np.argsort(angles)
    angles = angles[idx]
    energies = energies[idx]

    # --- periodic extension ---
    energies_ext = np.concatenate([energies, energies])

    # --- derivative sign ---
    dE = np.diff(energies_ext)
    sign = np.sign(dE)
    sign[sign == 0] = 1

    minima_idx = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0] + 1
    maxima_idx = np.where((sign[:-1] > 0) & (sign[1:] < 0))[0] + 1

    n = len(angles)
    minima_idx = minima_idx[minima_idx < n]
    maxima_idx = maxima_idx[maxima_idx < n]

    # --- filter minima by energy ---
    if energy_window is not None and len(minima_idx) > 0:
        E0 = energies.min()
        keep = energies[minima_idx] <= E0 + energy_window
        minima_idx = minima_idx[keep]

    minima_idx = np.sort(minima_idx)

    minima = [
        {"angle": float(angles[i]), "energy": float(energies[i]), "idx": int(i)} for i in minima_idx
    ]

    # --- compute barriers ---
    barriers: List[Dict[str, Any]] = []

    MIN_MINIMA = 2
    if len(minima_idx) >= MIN_MINIMA:
        for k in range(len(minima_idx)):
            i1 = minima_idx[k]
            i2 = minima_idx[(k + 1) % len(minima_idx)]

            if i2 > i1:
                segment = np.arange(i1, i2 + 1)
            else:
                segment = np.concatenate([np.arange(i1, n), np.arange(0, i2 + 1)])

            seg_E = energies[segment]
            seg_A = angles[segment]

            j = np.argmax(seg_E)

            ts_energy = float(seg_E[j])
            ts_angle = float(seg_A[j])

            barriers.append(
                {
                    "min1_angle": float(angles[i1]),
                    "min2_angle": float(angles[i2]),
                    "ts_angle": ts_angle,
                    "ts_energy": ts_energy,
                    "barrier_from_min1": ts_energy - float(energies[i1]),
                    "barrier_from_min2": ts_energy - float(energies[i2]),
                    "angle_delta": abs((float(angles[i2]) - float(angles[i1]) + 180) % 360 - 180),
                }
            )

    min_barrier = min(b["barrier_from_min1"] for b in barriers) if barriers else 0.0

    MAX_ANGLE_DELTA = 180
    return {
        "minima": minima,
        "barriers": barriers,
        "min_barrier": min_barrier,
        "angle_deltas": [b["angle_delta"] for b in barriers if b["angle_delta"] <= MAX_ANGLE_DELTA],
    }


def run_torsion_scan_omm(
    rdmol: Chem.Mol,
    bond_indices: tuple[int, int],
    steps: int = 24,
    relax_fmax: float = 0.5,
) -> float | tuple[float, list[float], list[float]]:
    """
    Perform a relaxed 1D torsion scan around the specified bond using OpenMM.

    Uses SMIRNOFF parameters and GBSA-OBC2 implicit solvent.
    """
    b_idx, c_idx = bond_indices
    # RDKit method to find neighbours for dihedral
    a_idx = None
    b_atom = rdmol.GetAtomWithIdx(b_idx)
    for nbr in b_atom.GetNeighbors():
        if nbr.GetIdx() != c_idx:
            a_idx = nbr.GetIdx()
            break

    d_idx = None
    c_atom = rdmol.GetAtomWithIdx(c_idx)
    for nbr in c_atom.GetNeighbors():
        if nbr.GetIdx() != b_idx:
            d_idx = nbr.GetIdx()
            break

    if a_idx is None or d_idx is None:
        logger.warning(
            f"Could not construct a rigid 4-atom dihedral constraint for bond {bond_indices}."
        )
        return 0.0, [], []

    offmol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
    forcefield = get_ligand_forcefield(offmol)

    top = offmol.to_topology().to_openmm()
    system = forcefield.createSystem(
        top,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )

    # Add Harmonic Torsion Restraint
    k = 1000 * unit.kilocalories_per_mole / unit.radians**2
    restraint = CustomTorsionForce(
        "0.5*k*dtheta^2; dtheta = min(diff, 2*pi-diff); diff = abs(theta - theta0);"
    )
    restraint.addGlobalParameter("k", k)
    restraint.addGlobalParameter("theta0", 0.0)
    restraint.addGlobalParameter("pi", math.pi)
    restraint.addTorsion(a_idx, b_idx, c_idx, d_idx)
    system.addForce(restraint)

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )
    sim = Simulation(top, system, integrator)
    sim.context.setPositions((offmol.conformers[0].m * unit.angstrom).in_units_of(unit.nanometer))

    angles = np.linspace(-180, 180, steps, endpoint=False)
    energies = []

    logger.info(
        f"Scanning dihedral path [{a_idx}-{b_idx}-{c_idx}-{d_idx}] via OpenMM constraint relax."
    )

    for angle in angles:
        theta0 = angle * math.pi / 180.0
        sim.context.setParameter("theta0", theta0)

        try:
            # minimizeEnergy tolerance is in kJ/mol/nm
            sim.minimizeEnergy(
                tolerance=relax_fmax * unit.kilojoules_per_mole / unit.nanometers,
                maxIterations=1000,
            )
            state = sim.context.getState(getEnergy=True)
            e_kcal = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            energies.append(e_kcal)
        except Exception as e:
            logger.warning(f"Optimization collapsed at {angle} deg: {e}")
            energies.append(float("nan"))

    energies = np.array(energies)
    energies = energies - energies.min()

    # Clean output and extrapolate
    valid_data = [(a, e) for a, e in zip(angles, energies, strict=False) if not np.isnan(e)]
    if not valid_data:
        return (0.0, [], [])

    valid_angles = [d[0] for d in valid_data]
    valid_energies = [d[1] for d in valid_data]

    result = torsion_barriers(valid_angles, valid_energies)

    # Group barriers by sorted minima pair; keep only the lowest ts_energy
    # (the kinetically relevant pathway between each pair of minima).
    best_barriers: dict[tuple[float, float], dict] = {}
    for barrier in result["barriers"]:
        key = tuple(sorted((barrier["min1_angle"], barrier["min2_angle"])))
        if key not in best_barriers or barrier["ts_energy"] < best_barriers[key]["ts_energy"]:
            best_barriers[key] = barrier

    return best_barriers, valid_angles, energies


def autodetect_flip_dihedrals_named(
    rdmol: Chem.Mol,
) -> list[tuple[str, str]]:
    """Like :func:`autodetect_flip_dihedrals` but returns PDB atom-name pairs.

    Each input bond is mapped from rdkit atom index to the atom's
    ``GetPDBResidueInfo().GetName()`` (stripped of whitespace), so the
    returned list is directly usable as a ``TitratableResidueReference.ring_flip_bonds``
    value (which keys atoms by name, not by index, for stability across
    OpenMM's ``Modeller.addHydrogens`` and rdkit/OpenFF conversions).

    ``rdmol`` must carry PDB residue info on every flagged atom; this is
    the case after
    :func:`opensqm.cph.reference_energy.protonation_states._assign_ligand_atom_names`
    (which :func:`opensqm.cph.reference_energy.protonation_states.build_protonation_states`
    calls internally) or :func:`opensqm.rdkit_utils.set_residue_info`.
    """
    bonds = autodetect_flip_dihedrals(rdmol)
    named: list[tuple[str, str]] = []
    for a, b in bonds:
        atom_a = rdmol.GetAtomWithIdx(int(a))
        atom_b = rdmol.GetAtomWithIdx(int(b))
        info_a = atom_a.GetPDBResidueInfo()
        info_b = atom_b.GetPDBResidueInfo()
        if info_a is None or info_b is None:
            missing = a if info_a is None else b
            raise ValueError(
                f"rdkit atom {missing} has no PDB residue info; call "
                f"_assign_ligand_atom_names or set_residue_info before "
                f"autodetect_flip_dihedrals_named"
            )
        named.append((info_a.GetName().strip(), info_b.GetName().strip()))
    return named


def autodetect_flip_dihedrals(
    rdmol: Chem.Mol,
) -> set[tuple[int, int]]:
    """Orchestrates the detection of Type 2 atropisomers that require MC flipping (using OpenMM)."""
    candidate_bonds = get_rotatable_terminal_bonds(rdmol)
    identified_flips = set()

    if not candidate_bonds:
        return []

    print(candidate_bonds)

    logger.info(f"Identified {len(candidate_bonds)} potential rotatable topology seeds.")

    for bond in candidate_bonds:
        best_barriers, angles, _energies = run_torsion_scan_omm(rdmol, bond)
        if not angles:
            continue

        MIN_TS_ENERGY = 5
        MAX_TS_ENERGY = 30

        print(best_barriers)

        for barrier in best_barriers.values():
            if MIN_TS_ENERGY < barrier["ts_energy"] < MAX_TS_ENERGY:
                logger.info(
                    f"--> [SUCCESS] Type 2 isolated! Adding {bond} to MC rotation handler list."
                )
                identified_flips.add(tuple(sorted(bond)))
            else:
                logger.info(
                    "--> [SKIP] Fluctuation rate not bound to simulation bottleneck regimes."
                )

    return identified_flips


if __name__ == "__main__":
    # rdmol = Chem.MolFromMolFile("data/inputs/PL-REX/003-CK2/1ZOE.sdf")
    rdmol = Chem.MolFromSmiles("c1cccc(Cl)[1c]1-[1c]1c(Br)cccc1")
    rdmol = Chem.AddHs(rdmol, addCoords=True)
    AllChem.EmbedMolecule(rdmol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rdmol)

    identified_flips = autodetect_flip_dihedrals(rdmol)
    print(identified_flips)
