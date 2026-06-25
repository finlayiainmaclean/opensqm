"""Module containing vanilla MD protocols."""

import copy
import time
from pathlib import Path

import numpy as np
from loguru import logger
from mdtraj.reporters import DCDReporter
from openmm import (
    LangevinMiddleIntegrator,
    State,
    System,
    app,
    unit,
)
from openmm.app.forcefield import ForceField
from openmm.app.topology import Topology
from pydantic import BaseModel, ConfigDict
from pydantic_units import OpenMMQuantity
from tqdm import tqdm

from opensqm.md.prepare import create_integrator, create_system
from opensqm.md.restraints import add_distal_restraints, add_restraints
from opensqm.md.terminal_ring_mc import TerminalRingMC, find_terminal_group


class ProductionSettings(BaseModel):
    """Settings for production MD."""

    model_config = ConfigDict(frozen=True)
    integrator_step_size: OpenMMQuantity[unit.picosecond] = 0.004 * unit.picoseconds
    log_interval: OpenMMQuantity[unit.picosecond] = 1 * unit.picoseconds
    run_time: OpenMMQuantity[unit.picosecond] = 0.5 * unit.nanoseconds
    rest_ligand: bool = True


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
    config: ProductionSettings,
    terminal_dihedrals: list[tuple[int, int]] | None = None,
) -> State:
    """
    Run production MD simulation with restraints.

    Args:
        topology: OpenMM topology
        positions: Equilibrated positions
        forcefield: OpenMM forcefield
        traj_path: Path to save trajectory (NetCDF format)
        config: Production simulation settings
        terminal_dihedrals: Optional terminal ring dihedrals for MC flips
    """
    num_steps_per_log = int(config.log_interval / config.integrator_step_size)
    steps = int(config.run_time / config.integrator_step_size)

    system = create_system(forcefield, topology, rest_ligand=config.rest_ligand)

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

    integrator = create_integrator(config.integrator_step_size)
    simulation = app.Simulation(topology, system, integrator)

    # Set box vectors before positions
    simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    simulation.context.setPositions(positions)

    # Quick minimization with production system
    simulation.minimizeEnergy()

    # Set velocities at production temperature
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

    # Add trajectory reporter
    reporter = DCDReporter(str(traj_path), num_steps_per_log)
    simulation.reporters.append(reporter)

    # Initialize TerminalRingMC if dihedrals are provided
    flipper = None

    if terminal_dihedrals:
        angles = list(np.arange(30, 210, 30))
        terminal_list = []
        for bond in terminal_dihedrals:
            terminal_list.append(find_terminal_group(topology, int(bond[0]), int(bond[1]), angles=angles))

        k_bt = 300 * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R
        flipper = TerminalRingMC(
            simulation=simulation,
            topology=topology,
            k_bt=k_bt,
            terminal_list=terminal_list,
        )

    # Production run with progress bar
    steps_per_update = num_steps_per_log
    total_updates = steps // steps_per_update
    ps_per_update = steps_per_update * config.integrator_step_size

    with tqdm(total=total_updates, desc="Production", unit="frames") as pbar:
        for i in range(total_updates):
            iter_start = time.time()

            # Attempt flips before progressing simulation block
            if flipper is not None:
                for _ in range(len(terminal_dihedrals) * 2):
                    flipper.move_dihe()

                if i % 100 == 0:
                    logger.info(
                        "TerminalRingMC: acceptance {rate:.1%} ({acc}/{att})",
                        rate=flipper.acceptance_rate,
                        acc=flipper.n_accepted,
                        att=flipper.n_attempts,
                    )




            simulation.step(steps_per_update)

            # Calculate performance
            iter_time = time.time() - iter_start
            sim_time_ns = ps_per_update / 1000.0
            ns_per_day = sim_time_ns * 86400 / iter_time if iter_time > 0 else 0

            # Get current simulation time
            current_time_ps = (i + 1) * ps_per_update


            pbar.set_postfix({"Time": f"{current_time_ps:.0f}ps", "ns/day": f"{ns_per_day:.2f}"})
            pbar.update(1)

    # Close reporters to prevent file handle leaks and double-free segfaults
    # when subsequent tools read/write to the same DCD file
    for r in simulation.reporters:
        try:
            r.close()
        except Exception:
            pass
    simulation.reporters.clear()

    # Optionally return final state
    final_state = simulation.context.getState(getPositions=True, getEnergy=True)

    return final_state
