import copy
import logging
import math
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from aimnet.calculators import AIMNet2ASE
from ase import Atoms
from ase.constraints import FixInternals
from ase.optimize import LBFGS
from openff.toolkit.topology import Molecule
from openff.toolkit.utils.toolkits import AmberToolsToolkitWrapper
from openmm import CustomTorsionForce, LangevinMiddleIntegrator, app, unit
from openmm.app import ForceField, Simulation
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from rdkit import Chem

logger = logging.getLogger(__name__)

# Base constants for Eyring
KB_T_OVER_H = 6.2e12  # k_B * T / h at 300K in s^-1
R_KCAL = 0.001987  # kcal / (mol * K)
T_K = 300.0  # K
EV_2_KCAL = 23.0605


def get_half_life(barrier_kcal: float, temp_k: float = T_K) -> float:
    """Calculates half life in seconds for a given barrier height in kcal/mol."""
    rate = KB_T_OVER_H * np.exp(-barrier_kcal / (R_KCAL * temp_k))
    return np.log(2) / rate


def get_barrier_kcal(half_life: float, temp_k: float = T_K) -> float:
    """Calculates barrier height in kcal/mol for a given half life in seconds."""
    rate = np.log(2) / half_life
    return -R_KCAL * temp_k * np.log(rate / KB_T_OVER_H)


def is_type_2_atropisomer(barrier_kcal: float, min_hl: float = 1e-9, max_hl: float = 2.5e6) -> bool:
    """
    Checks if a barrier falls within the Type 2 atropisomer timeframe.
    min_hl: 1 ns (too slow for MD to capture)
    max_hl: ~1 month (will cross organically during a drug binding assay but not stable indefinitely)
    """
    hl = get_half_life(barrier_kcal)
    return min_hl < hl < max_hl


def get_rotatable_terminal_bonds(rdmol: Chem.Mol) -> list[tuple[int, int]]:
    """
    Finds rotatable bonds conceptually matching ring attachments.
    Uses SMARTs to identify acyclic single bonds between distinct heavy-atom cores.
    """
    # Looks for any single acyclic bond! You can prune this SMARTs further
    # to target exact C2-symmetric functional groups like rings: '[!R]-!@[c1ccccc1]'
    rotatable_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")
    matches = rdmol.GetSubstructMatches(rotatable_smarts)

    bonds = []
    # Avoid duplicates
    for m in matches:
        pair = tuple(sorted([m[0], m[1]]))
        if pair not in bonds:
            bonds.append(pair)

    return bonds


def build_ase_atoms(rdmol: Chem.Mol, conf_id: int = 0) -> Atoms:
    """Convert RDKit Mol directly into an ASE Atoms sequence."""
    conf = rdmol.GetConformer(conf_id)
    positions = conf.GetPositions()
    atom_symbols = [atom.GetSymbol() for atom in rdmol.GetAtoms()]

    return Atoms(symbols=atom_symbols, positions=positions)


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
        Apply Savitzky–Golay smoothing before extrema detection.
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

    if len(minima_idx) >= 2:
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
                    "angle_delta": abs(float(angles[i1]) - float(angles[i2])),
                }
            )

    min_barrier = min(b["barrier_from_min1"] for b in barriers) if barriers else 0.0

    return {
        "minima": minima,
        "barriers": barriers,
        "min_barrier": min_barrier,
        "angle_deltas": [b["angle_delta"] for b in barriers if b["angle_delta"] <= 180],
    }


def atoms_from_mol(rdmol: Chem.Mol, conf_id: int = 0) -> Atoms:
    """Convert RDKit Mol directly into an ASE Atoms sequence with formal charges."""
    conf = rdmol.GetConformer(conf_id)
    positions = conf.GetPositions()
    atom_symbols = [atom.GetSymbol() for atom in rdmol.GetAtoms()]
    formal_charges = [atom.GetFormalCharge() for atom in rdmol.GetAtoms()]

    atoms = Atoms(symbols=atom_symbols, positions=positions)
    atoms.set_initial_charges(formal_charges)
    return atoms


def run_torsion_scan_ase(
    mol: Chem.Mol,
    bond_indices: tuple[int, int],
    steps: int = 24,
    relax_fmax: float = 0.05,
) -> float | tuple[float, list[float], list[float]]:
    """
    Performs a relaxed 1D torsion scan around the specified bond using AIMNet2.
    Returns the maximum barrier height in kcal/mol relative to the global minima.
    """
    atoms = atoms_from_mol(mol)

    total_charge = sum(atoms.get_initial_charges())
    calc = AIMNet2ASE("aimnet2", charge=total_charge)

    b_idx, c_idx = bond_indices
    a_idx, d_idx = None, None

    # Establish the full 4-coordinate dihedral pathway a-b-c-d securely
    # Safely sort local distances to find the immediate covalently bonded neighbour
    dists_b = np.linalg.norm(atoms.positions - atoms.positions[b_idx], axis=1)
    for idx in np.argsort(dists_b):
        if idx != b_idx and idx != c_idx:
            a_idx = int(idx)
            break

    dists_c = np.linalg.norm(atoms.positions - atoms.positions[c_idx], axis=1)
    for idx in np.argsort(dists_c):
        if idx != c_idx and idx != b_idx:
            d_idx = int(idx)
            break

    if a_idx is None or d_idx is None:
        logger.warning(
            f"Could not construct a rigid 4-atom dihedral constraint for bond {bond_indices}."
        )
        return 0.0

    energies = []
    angles = np.linspace(-180, 180, steps, endpoint=False)
    current_atoms = copy.deepcopy(atoms)
    current_atoms.calc = calc

    logger.info(
        f"Scanning dihedral path [{a_idx}-{b_idx}-{c_idx}-{d_idx}] via AIMnet2 constraint relaxations."
    )

    for angle in angles:
        # Snap the internal coordinates smoothly mapping the geometry
        current_atoms.set_dihedral(a_idx, b_idx, c_idx, d_idx, angle)

        # Lock only this dihedral for the optimizer
        dihedral_constraint = FixInternals(dihedrals_deg=[(angle, [a_idx, b_idx, c_idx, d_idx])])
        current_atoms.set_constraint(dihedral_constraint)

        opt = LBFGS(current_atoms, logfile=None)
        try:
            opt.run(fmax=relax_fmax, steps=100)
            # Energy naturally resolved in eV
            e_ev = current_atoms.get_potential_energy() * EV_2_KCAL
            energies.append(e_ev)
        except Exception as e:
            logger.warning(f"Optimization collapsed at {angle} deg: {e}")
            energies.append(float("nan"))

    energies = np.array(energies)
    energies = energies - energies.min()

    # Clean output and extrapolate
    valid_data = [(a, e) for a, e in zip(angles, energies) if not np.isnan(e)]
    if not valid_data:
        return (0.0, [], [])

    valid_angles = [d[0] for d in valid_data]
    valid_energies = [d[1] for d in valid_data]

    result = torsion_barriers(valid_angles, valid_energies)

    return result, valid_angles, energies


def run_torsion_scan_omm(
    rdmol: Chem.Mol,
    bond_indices: tuple[int, int],
    steps: int = 24,
    relax_fmax: float = 0.5,
) -> float | tuple[float, list[float], list[float]]:
    """
    Performs a relaxed 1D torsion scan around the specified bond using OpenMM
    with SMIRNOFF parameters and GBSA-OBC2 implicit solvent.
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
    offmol.assign_partial_charges("am1bcc", toolkit_registry=AmberToolsToolkitWrapper())

    forcefield = ForceField("implicit/obc1.xml")
    smirnoff = SMIRNOFFTemplateGenerator(
        forcefield="openff-2.2.0.offxml", molecules=offmol, cache="smirnoff.json"
    )
    forcefield.registerTemplateGenerator(smirnoff.generator)

    top = offmol.to_topology().to_openmm()
    system = forcefield.createSystem(
        top,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )

    # # Add Implicit Solvent manually mapping elements to OBC2 mbondi2 radii
    # radii = {
    #     1: (0.12, 0.85),
    #     6: (0.17, 0.72),
    #     7: (0.155, 0.79),
    #     8: (0.15, 0.85),
    #     9: (0.15, 0.88),
    #     15: (0.185, 0.86),
    #     16: (0.18, 0.96),
    #     17: (0.17, 0.8),
    # }
    # gbsa = GBSAOBCForce()
    # gbsa.setSolventDielectric(78.5)
    # gbsa.setSoluteDielectric(1.0)
    # for atom in offmol.atoms:
    #     rad, scale = radii.get(atom.atomic_number, (0.15, 0.8)) # default fallback
    #     gbsa.addParticle(atom.partial_charge.m, rad, scale)
    # system.addForce(gbsa)

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
        f"Scanning dihedral path [{a_idx}-{b_idx}-{c_idx}-{d_idx}] via OpenMM constraint relaxations."
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
    valid_data = [(a, e) for a, e in zip(angles, energies) if not np.isnan(e)]
    if not valid_data:
        return (0.0, [], [])

    valid_angles = [d[0] for d in valid_data]
    valid_energies = [d[1] for d in valid_data]

    result = torsion_barriers(valid_angles, valid_energies)

    return result, valid_angles, energies


def autodetect_flip_dihedrals(
    rdmol: Chem.Mol, method: Literal["openmm", "ase"] = "ase"
) -> list[tuple[int, int]]:
    """
    Orchestrates the detection of Type 2 atropisomers that require MC flipping (using OpenMM).
    """
    candidate_bonds = get_rotatable_terminal_bonds(rdmol)
    identified_flips = []

    if not candidate_bonds:
        return []

    logger.info(f"Identified {len(candidate_bonds)} potential rotatable topology seeds.")

    match method:
        case "openmm":
            func = run_torsion_scan_omm
        case "ase":
            func = run_torsion_scan_ase

    for bond in candidate_bonds:
        result, angles, energies = func(rdmol, bond)
        if not angles:
            continue

        print(result)

        for barrier in result["barriers"]:
            if barrier["ts_energy"] > 5 and barrier["ts_energy"] < 20:
                angle_delta = abs(barrier["min2_angle"] - barrier["min1_angle"])
                logger.info(
                    f"--> [SUCCESS] Type 2 kinetic block isolated! Adding {bond} to MC rotation handler list."
                )
                identified_flips.append((bond, angle_delta))
            else:
                logger.info(
                    "--> [SKIP] Fluctuation rate not bound to simulation bottleneck regimes."
                )

    return identified_flips
