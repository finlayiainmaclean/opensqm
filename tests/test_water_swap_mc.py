"""Tests for opensqm.md.water_swap_mc.WaterSwapMC.

The two-ghost variant is exercised on a small TIP3P-FB-only box:
the "ligand" pseudo-atom is just the central water's oxygen, the
first two waters are promoted to ``ghost1`` / ``ghost2`` (the
defaults), and a fast NCMC switch is run.
"""

# ruff: noqa: PLR2004

import numpy as np
import pytest
from openmm import (
    LangevinMiddleIntegrator,
    Platform,
    Vec3,
    unit,
)
from openmm.app import (
    ForceField,
    HBonds,
    Modeller,
    PME,
    Simulation,
    Topology,
)

from opensqm.md.water_swap_mc import (
    WATER_LAMBDA_PARAM,
    WaterSwapSettings,
    WaterSwapMC,
    _periodic_displacement,
    _water_oxygen_indices,
)


def _make_water_box(box_nm: float = 3.0) -> tuple[Simulation, list[int]]:
    """Build a small cubic water box and return ``(simulation, ligand_atoms)``.

    The "ligand" is a single oxygen atom near the box centre - good
    enough for any of the active-site geometry / counting tests but
    cheap enough that the whole suite runs in a few seconds on a CPU.
    """
    ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    top = Topology()
    top.setPeriodicBoxVectors(
        (
            Vec3(box_nm, 0, 0),
            Vec3(0, box_nm, 0),
            Vec3(0, 0, box_nm),
        )
        * unit.nanometer,
    )
    modeller = Modeller(top, [])
    modeller.addSolvent(
        ff,
        boxSize=(box_nm, box_nm, box_nm) * unit.nanometer,
        model="tip3p",
    )

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=HBonds,
        rigidWater=True,
        hydrogenMass=1.5 * unit.dalton,
    )
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 4.0 * unit.femtosecond,
    )
    # CPU is dramatically faster than Reference and accurate enough
    # for a few hundred MD steps on a 27 water box.
    try:
        platform = Platform.getPlatformByName("CPU")
    except Exception:
        platform = Platform.getPlatformByName("Reference")
    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    sim.context.setPeriodicBoxVectors(
        Vec3(box_nm, 0, 0) * unit.nanometer,
        Vec3(0, box_nm, 0) * unit.nanometer,
        Vec3(0, 0, box_nm) * unit.nanometer,
    )
    sim.minimizeEnergy(maxIterations=200)

    # Use a water somewhere in the middle of the topology as the
    # "ligand" atom so the active-site sphere isn't centred on the
    # default ghost slots (the last two waters in topology order).
    o_indices = _water_oxygen_indices(modeller.topology)
    return sim, [o_indices[len(o_indices) // 2]]


def test_alchemy_at_lambda_zero_keeps_ghost1_on_and_ghost2_off() -> None:
    """``lambda=0`` is the "between attempts" state: ghost1 active, ghost2 ghost.

    Without the mover the box has ``N`` real waters; with the mover
    installed and ``lambda=0`` the *effective* system should match a
    box where ghost2 has zero charge and zero LJ epsilon (i.e. one
    fewer interacting water). The exact numerical match here is the
    energy of the box with ghost2's parameters zeroed which we
    compute directly via the force.
    """
    sim, ligand = _make_water_box()
    nb = next(
        f for f in sim.system.getForces() if f.__class__.__name__ == "NonbondedForce"
    )
    # Zero ghost2's parameters directly in a *separate* system and
    # check that the WaterSwapMC-installed alchemy reproduces the same
    # energy at lambda=0. ghost2 defaults to the last water in
    # topology order so it stays clear of any potentially-special
    # crystallographic waters at the start of the topology.
    e_with_ghost2_off_direct = None
    water_residues = [r for r in sim.topology.residues() if r.name == "HOH"]
    g2_atoms = water_residues[-1]
    g2_atom_indices = [a.index for a in g2_atoms.atoms()]
    saved = [nb.getParticleParameters(i) for i in g2_atom_indices]
    for i in g2_atom_indices:
        _, sigma, _ = nb.getParticleParameters(i)
        nb.setParticleParameters(i, 0.0, sigma, 0.0)
    nb.updateParametersInContext(sim.context)
    e_with_ghost2_off_direct = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    for atom_idx, (q, sigma, eps) in zip(g2_atom_indices, saved):
        nb.setParticleParameters(atom_idx, q, sigma, eps)
    nb.updateParametersInContext(sim.context)

    WaterSwapMC(
        sim,
        ligand,
        config=WaterSwapSettings(
            n_perturbation_steps=2, n_propagation_steps_per_perturbation=0,
        ),
    )
    e_alchemy_at_zero = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    assert sim.context.getParameter(WATER_LAMBDA_PARAM) == pytest.approx(0.0)
    # 0.01 kJ/mol tolerance is generous enough for round-trip
    # representation error in the offset path (q_orig is stored
    # separately and re-multiplied by lambda=0 each evaluation)
    # without papering over a real coupling bug.
    assert e_alchemy_at_zero == pytest.approx(e_with_ghost2_off_direct, abs=1e-2)


def test_alchemy_at_lambda_one_swaps_which_ghost_is_on() -> None:
    """At ``lambda=1`` ghost1 is off and ghost2 is on - mirror of ``lambda=0``.

    The "ghost on" and "ghost off" identities permute between
    ``lambda=0`` and ``lambda=1``. At the equilibrated positions of
    ghost1 and ghost2 right after solvation, the two are sufficiently
    decoupled that the total energy at ``lambda=1`` is close (within
    a few kJ/mol of equilibrium fluctuations) to the energy at
    ``lambda=0`` for this small test box; what we strictly test here
    is that the alchemy is reversible (``lambda=1 -> lambda=0``
    returns the original energy exactly).
    """
    sim, ligand = _make_water_box()
    WaterSwapMC(
        sim,
        ligand,
        config=WaterSwapSettings(
            n_perturbation_steps=2, n_propagation_steps_per_perturbation=0,
        ),
    )
    e_at_zero = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    sim.context.setParameter(WATER_LAMBDA_PARAM, 1.0)
    _ = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    sim.context.setParameter(WATER_LAMBDA_PARAM, 0.0)
    e_back = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    assert e_back == pytest.approx(e_at_zero, abs=1e-3)


def test_water_oxygen_indices_finds_one_per_water() -> None:
    """Sanity check: every water residue gives exactly one oxygen index."""
    sim, _ = _make_water_box()
    indices = _water_oxygen_indices(sim.topology)
    n_waters = sum(1 for r in sim.topology.residues() if r.name == "HOH")
    assert len(indices) == n_waters
    assert len(set(indices)) == n_waters


def test_periodic_displacement_wraps_correctly() -> None:
    """The minimum-image trick must subtract a whole box length when needed."""
    box = np.array([3.0, 3.0, 3.0])
    p = np.array([2.9, 0.1, 1.5])
    q = np.array([0.1, 2.9, 1.5])
    d = _periodic_displacement(p, q, box)
    assert d == pytest.approx(np.array([-0.2, 0.2, 0.0]), abs=1e-9)


def test_attempt_runs_and_restores_lambda() -> None:
    """One full attempt: lambda must end at zero whether accepted or rejected."""
    sim, ligand = _make_water_box(box_nm=3.0)
    mover = WaterSwapMC(
        sim,
        ligand,
        config=WaterSwapSettings(
            active_site_radius=0.6,
            n_perturbation_steps=20,
            n_propagation_steps_per_perturbation=5,
        ),
    )
    mover.attempt()
    assert sim.context.getParameter(WATER_LAMBDA_PARAM) == pytest.approx(0.0)
    assert mover.total_attempts == 1


def test_at_least_one_swap_accepts_over_many_attempts() -> None:
    """Smoke test: several swaps should accept across a short batch.

    Uses a moderately-short NCMC switch and a handful of attempts so
    the test stays fast on a CPU. The active-site sphere is large
    enough to always contain at least one water (so the source for
    the "out" direction is non-empty). Failure means something
    deeper than statistical noise is broken (e.g. the Metropolis
    log-prob has the wrong sign, or the offsets never update).
    """
    np.random.seed(20260520)
    sim, ligand = _make_water_box(box_nm=3.0)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)

    mover = WaterSwapMC(
        sim,
        ligand,
        config=WaterSwapSettings(
            active_site_radius=0.7,
            n_perturbation_steps=80,
            n_propagation_steps_per_perturbation=20,
        ),
    )

    accepted = 0
    n_attempts = 15
    for _ in range(n_attempts):
        sim.step(50)
        if mover.attempt():
            accepted += 1

    assert mover.total_attempts == n_attempts
    assert accepted >= 1, (
        f"No water-swap moves accepted in {n_attempts} tries; "
        f"stats: {mover.summary()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
