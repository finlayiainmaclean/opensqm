# ruff: noqa: D100, D103, E501
import copy
from collections.abc import Generator, Sequence
from typing import Literal

import numpy as np
from openmm import (  # type: ignore
    CustomExternalForce,
    unit,
)
from openmm.app.topology import Atom  # type: ignore
from openmm.openmm import System  # type: ignore
from openmm.unit import Quantity  # type: ignore


def add_distal_restraints(
    system: System,
    positions: Quantity,
    atoms: Generator[Atom, None, None],
    min_distance: float = 0.8,
    max_distance: float = 1.5,
    max_restraint_force: float = 10.0,
    restraints: Sequence[Literal["ligand", "backbone", "heavy_atom", "protein", "solvent"]] = (
        "ligand",
        "backbone",
    ),
) -> System:
    # Add restraints to system for equilibration in place with distance-dependent force constants

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

    # Create force with per-particle force constant
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

            if min_distance_nm < 1.0:
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
    restraint_force: float,
    restraints: Sequence[Literal["ligand", "backbone", "heavy_atom", "protein", "solvent"]] = (
        "ligand",
        "backbone",
    ),
) -> System:
    # Add restrains to system for equilibration in place. The restraints are added to the backbone atoms of the protein, and heavy atoms of the ligand.
    force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force_amount = restraint_force * unit.kilocalories_per_mole / unit.angstroms**2
    force.addGlobalParameter("k", force_amount)
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
