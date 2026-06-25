# ruff: noqa: D100, D103, E501
import copy
from collections.abc import Generator, Sequence
from typing import Literal

import numpy as np
from openmm import (
    CustomExternalForce,
    unit,
)
from openmm.app.topology import Atom
from openmm.openmm import System
from openmm.unit import Quantity
from pydantic_units import OpenMMQuantity


def add_distal_restraints(
    system: System,
    positions: Quantity,
    atoms: Generator[Atom, None, None],
    min_distance: float = 0.8,
    max_distance: float = 1.2,
    max_restraint_force: float = 10.0,
    restraints: Sequence[Literal["ligand", "backbone", "heavy_atom", "protein", "solvent"]] = (
        "ligand",
        "backbone",
    ),
    flat_bottom_sigma: float = 0.0,
    exclusion_distance: float = 1.0,
    exclude_by_residue: bool = False,
) -> System:
    # Add restraints to system for equilibration in place with distance-dependent force constants.
    # ``flat_bottom_sigma`` (Angstrom): if > 0 use a flat-bottom restraint, no force until the atom
    # is displaced more than sigma from its reference position (paper backbone sigma = 3.0 A).
    # ``exclusion_distance`` (nm): atoms nearer the ligand than this are left free.
    # ``exclude_by_residue``: free the entire residue when any of its atoms is inside the exclusion
    # distance (paper: residues within 6 A of the binding site).

    # Convert atoms generator to list so we can iterate multiple times
    atoms_list = list(atoms)

    # First pass: identify LIG atoms and their positions
    lig_positions = []
    for i, atom in enumerate(atoms_list):
        if atom.residue.name == "LIG":
            lig_positions.append(positions[i].value_in_unit(unit.nanometers))

    if len(lig_positions) == 0:
        raise ValueError("No LIG residue found in system")

    lig_positions = np.array(lig_positions)

    # Optional: min distance from each residue to the binding site (over residue atoms).
    residue_min_distance: dict[object, float] = {}
    if exclude_by_residue:
        for atom_crd, atom in zip(positions, atoms_list, strict=False):
            atom_pos = atom_crd.value_in_unit(unit.nanometers)
            d = float(np.min(np.linalg.norm(lig_positions - atom_pos, axis=1)))
            if d < residue_min_distance.get(atom.residue, np.inf):
                residue_min_distance[atom.residue] = d

    # Create force with per-particle force constant
    if flat_bottom_sigma > 0.0:
        force = CustomExternalForce(
            "k*(max(0, periodicdistance(x, y, z, x0, y0, z0) - sigma))^2"
        )
        force.addGlobalParameter("sigma", flat_bottom_sigma / 10.0)  # Angstrom -> nm
    else:
        force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addPerParticleParameter("k")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    # Second pass: add restraints with distance-dependent force constants
    for i, (atom_crd, atom) in enumerate(zip(positions, atoms_list, strict=False)):
        atom_in_backbone = (atom.name in ("CA", "C", "N")) and (
            atom.residue.name not in ["LIG", "COF", "ACE", "NME"]
        )  # Backbone atoms
        is_ligand = atom.name[0] != "H" and (atom.residue.name in ["LIG", "COF"])
        heavy_atom = atom.name[0] != "H"

        is_solvent_or_ion = atom.residue.name in ["HOH", "SOL", "WAT", "NA", "CL", "MG", "K", "ZN"]

        add_atom = False
        if "ligand" in restraints and is_ligand and not is_solvent_or_ion:
            add_atom = True
        if "backbone" in restraints and atom_in_backbone and not is_solvent_or_ion:
            add_atom = True
        if "heavy_atom" in restraints and heavy_atom and not is_solvent_or_ion:
            add_atom = True
        if "protein" in restraints and heavy_atom and not is_solvent_or_ion and not is_ligand:
            add_atom = True

        if add_atom:
            # Calculate minimum distance to any LIG atom
            atom_pos = atom_crd.value_in_unit(unit.nanometers)
            distances = np.linalg.norm(lig_positions - atom_pos, axis=1)
            min_distance_nm = np.min(distances)

            if exclude_by_residue:
                if residue_min_distance.get(atom.residue, np.inf) < exclusion_distance:
                    continue
            elif min_distance_nm < exclusion_distance:
                continue

            # Calculate distance-dependent force constant using linear interpolation
            k_magnitude = np.interp(
                min_distance_nm, [min_distance, max_distance], [0.0, max_restraint_force]
            )
            k_value = k_magnitude * unit.kilocalories_per_mole / unit.angstroms**2

            force.addParticle(i, [k_value, *atom_crd.value_in_unit(unit.nanometers)])

    posres_sys = copy.deepcopy(system)
    posres_sys.addForce(force)
    restraint_force_idx = len(posres_sys.getForces()) - 1

    return posres_sys, restraint_force_idx


def add_restraints(
    system: System,
    positions: Quantity,
    atoms: Generator[Atom, None, None],
    restraint_force: OpenMMQuantity[unit.kilocalories_per_mole / unit.angstroms**2] = 4.0 * unit.kilocalories_per_mole / unit.angstroms**2,
    restraints: Sequence[Literal["ligand", "backbone", "heavy_atom", "protein", "solvent"]] = (
        "ligand",
        "backbone",
    ),
    periodic: bool = True,
) -> System:
    # Add restrains to system for equilibration in place. The restraints are added to the backbone atoms of the protein, and heavy atoms of the ligand.
    if periodic:
        expr = "k*periodicdistance(x, y, z, x0, y0, z0)^2"
    else:
        expr = "k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    force = CustomExternalForce(expr)
    force.addGlobalParameter("k", restraint_force)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for i, (atom_crd, atom) in enumerate(zip(positions, atoms, strict=False)):
        atom_in_backbone = (atom.name in ("CA", "C", "N")) and (
            atom.residue.name not in ["LIG", "COF", "ACE", "NME"]
        )  # Backbone atoms
        atom_in_ligand = atom.name[0] != "H" and (
            atom.residue.name in ["LIG", "COF"]
        )  # Heavy atoms in ligand
        heavy_atom = atom.name[0] != "H"

        is_solvent_or_ion = atom.residue.name in ["HOH", "SOL", "WAT", "NA", "CL", "MG", "K", "ZN"]

        add_atom = False
        if "solvent" in restraints and is_solvent_or_ion:
            add_atom = True
        if "ligand" in restraints and atom_in_ligand and not is_solvent_or_ion:
            add_atom = True
        if "backbone" in restraints and atom_in_backbone and not is_solvent_or_ion:
            add_atom = True
        if "heavy_atom" in restraints and heavy_atom and not is_solvent_or_ion:
            add_atom = True
        if add_atom:
            force.addParticle(i, atom_crd.value_in_unit(unit.nanometers))
    posres_sys = copy.deepcopy(system)
    posres_sys.addForce(force)
    restraint_force_idx = len(posres_sys.getForces()) - 1  # The last force is the restraint force

    return posres_sys, restraint_force_idx
