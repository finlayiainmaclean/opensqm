# ruff: noqa: D100, D103, E501
import copy
import io
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from openff.toolkit.topology import Molecule  # type: ignore
from openff.toolkit.utils.toolkits import AmberToolsToolkitWrapper  # type: ignore
from openmm import LangevinMiddleIntegrator, app, unit  # type: ignore
from openmm.app import Modeller, PDBFile  # type: ignore
from openmmforcefields.generators import SMIRNOFFTemplateGenerator  # type: ignore
from rdkit import Chem

from opensqm.md.restraints import add_distal_restraints, add_restraints
from opensqm.rdkit_utils import get_coordinates, set_coordinates

logging.getLogger("openff.interchange.smirnoff").setLevel(logging.WARNING)
logging.getLogger("openmmforcefields.generators.template_generators").setLevel(logging.WARNING)


def relax_complex(
    *, ligand: Chem.Mol, protein: Chem.Mol, simulation_time: float = 60
) -> tuple[Chem.Mol, Chem.Mol]:

    offmol = Molecule.from_rdkit(ligand, allow_undefined_stereo=False)
    offmol.assign_partial_charges("am1bcc", toolkit_registry=AmberToolsToolkitWrapper())

    smirnoff = SMIRNOFFTemplateGenerator(forcefield="openff-2.2.0.offxml", molecules=offmol)

    files = [
        "amber/ff14SB.xml",
        "amber/phosaa10.xml",
        "amber/tip3p_standard.xml",
        "implicit/gbn2.xml",
    ]

    forcefield = app.ForceField(*files)
    forcefield.registerTemplateGenerator(smirnoff.generator)

    protein_pdb = PDBFile(io.StringIO(Chem.MolToPDBBlock(protein)))

    lig_top = offmol.to_topology().to_openmm()
    lig_pos = (offmol.conformers[0].m * unit.angstrom).in_units_of(unit.nanometer)

    for chain in lig_top.chains():
        for res in chain.residues():
            res.name = "LIG"

    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    modeller.add(lig_top, lig_pos)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=12.0 * unit.angstroms,
        switchDistance=10.0 * unit.angstroms,
        constraints=app.HBonds,
        rigidWater=True,
    )

    positions = anneal_and_minimise(
        modeller.positions, modeller.topology, system, annealing_time=simulation_time
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        minimised_pdb_path = Path(tmpdir) / "com.pdb"
        # OpenFF RDKit import uses int residue numbers; PDBFile.writeFile(keepIds=True) requires str ids.
        for res in modeller.topology.residues():
            res.id = str(res.id)
        app.PDBFile.writeFile(
            modeller.topology,
            positions,
            open(str(minimised_pdb_path), "w"),
            keepIds=True,
        )
        complex_opt = Chem.MolFromPDBFile(
            str(minimised_pdb_path), removeHs=False, sanitize=False, proximityBonding=False
        )

    # Extract ligand and protein conformations from the optimised complex
    ligand_idxs = [
        atom.GetIdx()
        for atom in complex_opt.GetAtoms()
        if atom.GetPDBResidueInfo().GetResidueName() == "LIG"
    ]
    pocket_idxs = [
        atom.GetIdx()
        for atom in complex_opt.GetAtoms()
        if atom.GetPDBResidueInfo().GetResidueName() != "LIG"
    ]

    rw = Chem.RWMol(complex_opt)
    for idx in sorted(ligand_idxs, reverse=True):
        rw.RemoveAtom(idx)
    protein_opt = rw.GetMol()

    rw = Chem.RWMol(complex_opt)
    for idx in sorted(pocket_idxs, reverse=True):
        rw.RemoveAtom(idx)
    ligand_opt = rw.GetMol()

    ligand_coords = get_coordinates(ligand_opt)
    set_coordinates(ligand, coords=ligand_coords)

    protein_coords = get_coordinates(protein_opt)
    set_coordinates(protein, coords=protein_coords)

    return ligand, protein


def anneal_and_minimise(
    positions: unit.Quantity,
    topology: Any,
    system: Any,
    integrator_ps_per_step: float = 0.002,
    annealing_time: float = 60.0,
) -> unit.Quantity:
    system = copy.deepcopy(system)

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        integrator_ps_per_step * unit.picoseconds,
    )
    integrator0 = copy.deepcopy(integrator)
    RESTRAINTS = ("heavy_atom",)
    system0, _ = add_restraints(system, positions, topology.atoms(), 4.0, restraints=RESTRAINTS)
    system0, _ = add_distal_restraints(
        system0, positions, topology.atoms(), 4.0, restraints=RESTRAINTS
    )
    simulation = app.Simulation(topology, system0, integrator0)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(50 * unit.kelvin)

    annealing_steps = annealing_time / integrator_ps_per_step
    N = 50
    annealing_steps_per_iteration = int(annealing_steps // N)

    temps = list(np.linspace(50, 300, N // 2)) + list(np.linspace(300, 50, N // 2))

    for current_temperature in temps:
        integrator0.setTemperature(current_temperature * unit.kelvin)
        simulation.step(annealing_steps_per_iteration)

    positions = simulation.context.getState(getPositions=True).getPositions()

    # Minimise the system with restraints on the ligand and backbone atoms
    RESTRAINTS = ("ligand", "backbone", "solvent")
    system1, _ = add_restraints(system, positions, topology.atoms(), 4.0, restraints=RESTRAINTS)
    system1, _ = add_distal_restraints(
        system1, positions, topology.atoms(), 4.0, restraints=RESTRAINTS
    )
    integrator1 = copy.deepcopy(integrator)
    simulation = app.Simulation(topology, system1, integrator1)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    positions = simulation.context.getState(getPositions=True).getPositions()

    # Minimise the system without restraints
    integrator2 = copy.deepcopy(integrator)
    simulation = app.Simulation(topology, system, integrator2)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()

    positions = simulation.context.getState(getPositions=True).getPositions()

    return positions


if __name__ == "__main__":
    ligand = Chem.MolFromMolFile("/tmp/ligand.sdf", removeHs=False)
    protein = Chem.MolFromPDBFile("/tmp/protein.pdb", removeHs=False)
    ligand, protein = relax_complex(ligand=ligand, protein=protein, simulation_time=60)
    Chem.MolToPDBFile(protein, "/tmp/protein.pdb")
    Chem.MolToMolFile(ligand, "/tmp/ligand.sdf")
