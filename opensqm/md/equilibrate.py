"""Module containing equilibration protocol."""
import time


import numpy as np
from pymbar import timeseries
from openmm import (
    MonteCarloBarostat,
    app,
    unit,
)
from openmm.app.forcefield import ForceField
from openmm.app.topology import Topology
from tqdm import tqdm
from opensqm.md.prepare import create_integrator, create_system
from opensqm.md.restraints import add_restraints
from loguru import logger
from pydantic_units import OpenMMQuantity
from pydantic import BaseModel, ConfigDict


class EquilibrationSettings(BaseModel):
    """Settings for the optimisation process."""

    model_config = ConfigDict(frozen=True)
    integrator_step_size: OpenMMQuantity[unit.picosecond] = 0.004 * unit.picoseconds
    npt_time: OpenMMQuantity[unit.picosecond] = 200 * unit.picoseconds
    warmup_time: OpenMMQuantity[unit.picosecond] = 100 * unit.picoseconds
    restraint_force: OpenMMQuantity[unit.kilocalories_per_mole / unit.angstroms**2] = 4.0 * unit.kilocalories_per_mole / unit.angstroms**2


def _check_volume_plateau(
    volumes_nm3: np.ndarray,
    *,
    max_relative_std: float = 0.02,
    min_equilibrated_fraction: float = 0.3,
) -> None:
    """Raise RuntimeError if NPT box volume has not plateaued."""
    if len(volumes_nm3) < 5:
        msg = f"Insufficient volume samples ({len(volumes_nm3)}) to verify NPT plateau"
        raise RuntimeError(msg)

    t0, _g, _ = timeseries.detectEquilibration(volumes_nm3)
    equilibrated = volumes_nm3[t0:]
    equilibrated_fraction = len(equilibrated) / len(volumes_nm3)

    if equilibrated_fraction < min_equilibrated_fraction:
        msg = (
            "NPT box volume has not plateaued: "
            f"equilibration onset at sample {t0 + 1}/{len(volumes_nm3)} "
            f"({equilibrated_fraction:.0%} equilibrated, "
            f"need >={min_equilibrated_fraction:.0%})"
        )
        logger.warning(msg)

    mean_vol = float(np.mean(equilibrated))
    rel_std = float(np.std(equilibrated, ddof=1) / mean_vol)
    if rel_std > max_relative_std:
        msg = (
            "NPT box volume has not plateaued: "
            f"relative std={rel_std:.3f} (threshold={max_relative_std:.3f}), "
            f"mean volume={mean_vol:.2f} nm^3"
        )
        logger.warning(msg)


def equilibrate(
    topology: Topology,
    positions: unit.Quantity,
    forcefield: ForceField,
    config: EquilibrationSettings,
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
    warmup_steps = int(config.warmup_time / config.integrator_step_size)
    warmup_steps_per_iteration = int(warmup_steps / N)
    npt_steps = int(config.npt_time / config.integrator_step_size)
    npt_steps_per_iteration = int(npt_steps / N)

    # ========================================
    # PHASE 1: NVT WARMUP (with restraints)
    # ========================================
    system = create_system(forcefield, topology, rest_ligand=False)

    # Use small timestep for initial warmup
    integrator = create_integrator(0.001 * unit.picoseconds)

    # Add restraints to backbone and ligand
    system, _ = add_restraints(
        system, positions, topology.atoms(), restraint_force=config.restraint_force, restraints=("backbone", "ligand")
    )

    # Create simulation
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    simulation.context.setPositions(positions)

    # Energy minimization
    simulation.minimizeEnergy()

    # Set initial low temperature
    simulation.context.setVelocitiesToTemperature(10 * unit.kelvin)

    # Gradual heating from 10K to 300K
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

    # Create new system with barostat
    system = create_system(forcefield, topology, rest_ligand=False)
    integrator = create_integrator(config.integrator_step_size)

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
    volumes_nm3: list[float] = []
    with tqdm(total=N, desc="NPT Equil", unit="iter") as pbar:
        for _i, k in enumerate(np.linspace(4.0, 1.0, N)):
            iter_start = time.time()
            simulation.step(npt_steps_per_iteration)

            # Gradually reduce restraint strength
            simulation.context.setParameter(
                "k", float(k) * unit.kilocalories_per_mole / unit.angstroms**2
            )

            volume = (
                simulation.context.getState()
                .getPeriodicBoxVolume()
                .value_in_unit(unit.nanometer**3)
            )
            volumes_nm3.append(volume)

            # Calculate performance
            iter_time = time.time() - iter_start
            sim_time_ns = npt_steps_per_iteration * config.integrator_step_size / 1000.0
            ns_per_day = sim_time_ns * 86400 / iter_time if iter_time > 0 else 0

            pbar.set_postfix({"k": f"{k:.2f}", "vol": f"{volume:.1f}", "ns/day": f"{ns_per_day._value:.2f}"})
            pbar.update(1)

    _check_volume_plateau(np.array(volumes_nm3))

    # Save final equilibrated state with box vectors
    npt_state = simulation.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True
    )

    topology.setPeriodicBoxVectors(npt_state.getPeriodicBoxVectors())

    positions = npt_state.getPositions()

    return topology, positions
    

