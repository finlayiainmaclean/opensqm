"""Module containing vanilla MD protocols."""

import copy
import logging
import time
from pathlib import Path

import numpy as np
from mdtraj.reporters import DCDReporter  # type: ignore
from openff.toolkit.topology import Molecule  # type: ignore
from openff.toolkit.utils.toolkits import AmberToolsToolkitWrapper  # type: ignore
from openmm import (  # type: ignore
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    State,
    System,
    app,
    unit,
)
from openmm.app import Modeller  # type: ignore
from openmm.app.forcefield import ForceField  # type: ignore
from openmm.app.topology import Topology  # type: ignore
from openmmforcefields.generators import SMIRNOFFTemplateGenerator  # type: ignore
from rdkit import Chem
from tqdm import tqdm

from opensqm.md.restraints import add_distal_restraints, add_restraints

logging.getLogger("openff.interchange.smirnoff").setLevel(logging.WARNING)


def create_system(forcefield: ForceField, topology: Topology) -> System:
    """Create an OpenMM System from the forcefield and topology."""
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=12.0 * unit.angstroms,
        switchDistance=10.0 * unit.angstroms,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=1.5,
    )
    return system


def create_integrator(integrator_ps_per_step: float) -> LangevinMiddleIntegrator:
    """Create a Langevin Middle Integrator."""
    return LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        integrator_ps_per_step * unit.picoseconds,
    )


def prepare_complex(
    ligand: Chem.Mol, protein_modeller: Modeller
) -> tuple[Topology, unit.Quantity, ForceField]:
    """Prepare the complex by building ligand and protein into a modeller."""
    files = [
        "amber/ff14SB.xml",
        "amber/phosaa10.xml",
        "amber/tip3p_HFE_multivalent.xml",
        "amber/tip3p_standard.xml",
    ]

    forcefield = app.ForceField(*files)

    offmol = Molecule.from_rdkit(ligand, allow_undefined_stereo=False)
    offmol.assign_partial_charges("am1bcc", toolkit_registry=AmberToolsToolkitWrapper())

    smirnoff = SMIRNOFFTemplateGenerator(
        forcefield="openff-2.2.0.offxml", molecules=offmol, cache="smirnoff.json"
    )
    forcefield.registerTemplateGenerator(smirnoff.generator)

    lig_top = offmol.to_topology().to_openmm()
    lig_pos = (offmol.conformers[0].m * unit.angstrom).in_units_of(unit.nanometer)

    for chain in lig_top.chains():
        for res in chain.residues():
            res.name = "LIG"

    modeller = Modeller(protein_modeller.topology, protein_modeller.positions)
    modeller.add(lig_top, lig_pos)

    modeller.addSolvent(
        forcefield,
        ionicStrength=0.15 * unit.molar,
        padding=1.0 * unit.nanometers,
        boxShape="cube",
        positiveIon="Na+",
        negativeIon="Cl-",
    )

    return modeller.topology, modeller.positions, forcefield


def equilibrate(
    topology: Topology,
    positions: unit.Quantity,
    forcefield: ForceField,
    integrator_ps_per_step: float = 0.004,
    npt_ps: float = 200,
    warmup_ps: float = 100,
) -> tuple[Topology, unit.Quantity]:
    """
    Equilibrate a molecular system through NVT warmup and NPT equilibration.

    Args:
        topology: OpenMM topology
        positions: Initial positions
        forcefield: OpenMM forcefield
        integrator_ps_per_step: Timestep in picoseconds (default: 0.004 ps = 4 fs)
        npt_ps: NPT equilibration time in picoseconds
        warmup_ps: NVT warmup time in picoseconds

    Returns
    -------
        tuple: (equilibrated_positions, equilibrated_box_vectors)
    """
    N = 30
    warmup_steps = int(warmup_ps / integrator_ps_per_step)
    warmup_steps_per_iteration = int(warmup_steps / N)
    npt_steps = int(npt_ps / integrator_ps_per_step)
    npt_steps_per_iteration = int(npt_steps / N)

    # ========================================
    # PHASE 1: NVT WARMUP (with restraints)
    # ========================================
    print("Setting up NVT warmup...")
    system = create_system(forcefield, topology)

    # Use small timestep for initial warmup
    integrator = create_integrator(0.001)

    # Add restraints to backbone and ligand
    system, _ = add_restraints(
        system, positions, topology.atoms(), 4.0, restraints=("backbone", "ligand")
    )

    # Create simulation
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    simulation.context.setPositions(positions)

    # Energy minimization
    print("Minimizing energy...")
    state = simulation.context.getState(getEnergy=True)
    print(f"Initial energy: {state.getPotentialEnergy()}")
    simulation.minimizeEnergy()
    state = simulation.context.getState(getEnergy=True)
    print(f"Minimized energy: {state.getPotentialEnergy()}")

    # Set initial low temperature
    simulation.context.setVelocitiesToTemperature(10 * unit.kelvin)

    # Gradual heating from 10K to 300K
    print("Starting NVT warmup phase...")
    with tqdm(total=N, desc="Warmup", unit="iter") as pbar:
        for _i, current_temperature in enumerate(np.linspace(10, 300, N)):
            iter_start = time.time()
            integrator.setTemperature(current_temperature * unit.kelvin)
            simulation.step(warmup_steps_per_iteration)

            # Calculate performance
            iter_time = time.time() - iter_start
            sim_time_ns = warmup_steps_per_iteration * 0.001 / 1000.0  # 0.001 ps timestep
            ns_per_day = sim_time_ns * 86400 / iter_time if iter_time > 0 else 0

            pbar.set_postfix({"Temp": f"{current_temperature:.1f}K", "ns/day": f"{ns_per_day:.2f}"})
            pbar.update(1)

    # Save warmup state including box vectors
    warmup_state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    warmup_positions = warmup_state.getPositions()
    warmup_box = warmup_state.getPeriodicBoxVectors()

    print(f"Warmup complete. Box vectors: {warmup_box}")

    # ========================================
    # PHASE 2: NPT EQUILIBRATION (gradually release restraints)
    # ========================================
    print("\nSetting up NPT equilibration...")

    # Create new system with barostat
    system = create_system(forcefield, topology)
    integrator = create_integrator(integrator_ps_per_step)

    # Add restraints (will be gradually reduced)
    system, _ = add_restraints(
        system, warmup_positions, topology.atoms(), 4.0, restraints=("backbone", "ligand")
    )

    # Add Monte Carlo barostat for NPT
    system.addForce(MonteCarloBarostat(1 * unit.atmosphere, 300 * unit.kelvin))

    # Create new simulation
    simulation = app.Simulation(topology, system, integrator)

    # CRITICAL: Set box vectors from warmup BEFORE setting positions
    simulation.context.setPeriodicBoxVectors(*warmup_box)
    simulation.context.setPositions(warmup_positions)

    # Quick minimization with new system
    simulation.minimizeEnergy()

    # Set velocities at target temperature
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

    # Add reporter to track density and volume
    density_reporter = app.StateDataReporter(
        "/tmp/npt_equilibration.log",
        100,
        step=True,
        volume=True,
        density=True,
        temperature=True,
        potentialEnergy=True,
    )
    simulation.reporters.append(density_reporter)

    # NPT equilibration with gradual restraint release
    print("Starting NPT equilibration...")
    with tqdm(total=N, desc="NPT Equil", unit="iter") as pbar:
        for _i, k in enumerate(np.linspace(4.0, 1.0, N)):
            iter_start = time.time()
            simulation.step(npt_steps_per_iteration)

            # Gradually reduce restraint strength
            simulation.context.setParameter(
                "k", float(k) * unit.kilocalories_per_mole / unit.angstroms**2
            )

            # Calculate performance
            iter_time = time.time() - iter_start
            sim_time_ns = npt_steps_per_iteration * integrator_ps_per_step / 1000.0
            ns_per_day = sim_time_ns * 86400 / iter_time if iter_time > 0 else 0

            pbar.set_postfix({"k": f"{k:.2f}", "ns/day": f"{ns_per_day:.2f}"})
            pbar.update(1)

    # Save final equilibrated state with box vectors
    npt_state = simulation.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True
    )

    final_box = npt_state.getPeriodicBoxVectors()
    print(f"\nEquilibration complete. Final box vectors: {final_box}")
    print(f"Final energy: {npt_state.getPotentialEnergy()}")

    topology.setPeriodicBoxVectors(npt_state.getPeriodicBoxVectors())

    positions = npt_state.getPositions()

    return topology, positions


def anneal_and_minimise(
    positions: unit.Quantity,
    topology: Topology,
    system: System,
    integrator_ps_per_step: float = 0.002,
    annealing_time: float = 60.0,
) -> unit.Quantity:
    """Anneal the system then minimize to find a nearby local minima."""
    system = copy.deepcopy(system)

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        integrator_ps_per_step * unit.picoseconds,
    )
    integrator0 = copy.deepcopy(integrator)
    system0, _ = add_restraints(
        system, positions, topology.atoms(), 4.0, restraints=("heavy_atom",)
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
    system1, _ = add_restraints(
        system,
        positions,
        topology.atoms(),
        4.0,
        restraints=("ligand", "backbone"),
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


def production(
    topology: Topology,
    positions: unit.Quantity,
    forcefield: ForceField,
    traj_path: str | Path,
    integrator_ps_per_step: float = 0.004,
    log_ps: int = 100,
    run_time_ps: int = 1000,
) -> State:
    """
    Run production MD simulation with restraints.

    Args:
        topology: OpenMM topology
        positions: Equilibrated positions
        forcefield: OpenMM forcefield
        traj_path: Path to save trajectory (NetCDF format)
        integrator_ps_per_step: Timestep in picoseconds
        log_ps: Logging interval in picoseconds
        run_time_ps: Total production time in picoseconds
    """
    num_steps_per_log = int(log_ps // integrator_ps_per_step)
    steps = int(run_time_ps / integrator_ps_per_step)

    print("Setting up production simulation...")
    system = create_system(forcefield, topology)

    # Add distal restraints to prevent protein unfolding
    add_distal_restraints(
        system,
        positions,
        topology.atoms(),
        min_distance=1.2,
        max_distance=1.5,
        restraints=("backbone",),
        max_restraint_force=10.0,
    )

    integrator = create_integrator(integrator_ps_per_step)
    simulation = app.Simulation(topology, system, integrator)

    # Set box vectors before positions
    print(topology.getPeriodicBoxVectors())
    simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    simulation.context.setPositions(positions)

    # Quick minimization with production system
    print("Minimizing energy...")
    simulation.minimizeEnergy()

    # Set velocities at production temperature
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

    # Add trajectory reporter
    reporter = DCDReporter(str(traj_path), num_steps_per_log)
    simulation.reporters.append(reporter)

    # Production run with progress bar
    print(f"Starting production run ({run_time_ps} ps)...")
    steps_per_update = num_steps_per_log
    total_updates = steps // steps_per_update

    with tqdm(total=total_updates, desc="Production", unit="frames") as pbar:
        for i in range(total_updates):
            iter_start = time.time()
            simulation.step(steps_per_update)

            # Calculate performance
            iter_time = time.time() - iter_start
            sim_time_ns = steps_per_update * integrator_ps_per_step / 1000.0
            ns_per_day = sim_time_ns * 86400 / iter_time if iter_time > 0 else 0

            # Get current simulation time
            current_time_ps = (i + 1) * steps_per_update * integrator_ps_per_step

            pbar.set_postfix({"Time": f"{current_time_ps:.0f}ps", "ns/day": f"{ns_per_day:.2f}"})
            pbar.update(1)

    print("Production simulation completed!")

    # Optionally return final state
    final_state = simulation.context.getState(getPositions=True, getEnergy=True)
    print(f"Final energy: {final_state.getPotentialEnergy()}")

    return final_state
