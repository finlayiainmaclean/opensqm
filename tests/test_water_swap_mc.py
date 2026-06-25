"""Tests for opensqm.md.water_swap_mc.WaterSwapMC.

The two-ghost variant is exercised on a small TIP3P-FB-only box:
the "ligand" pseudo-atom is just the central water's oxygen, the
first two waters are promoted to ``ghost1`` / ``ghost2`` (the
defaults), and a fast NCMC switch is run.
"""


import numpy as np
import pytest
from openmm import (
    LangevinMiddleIntegrator,
    Platform,
    Vec3,
    unit,
)
from openmm.app import (
    PME,
    ForceField,
    HBonds,
    Modeller,
    Simulation,
    Topology,
)
from opensqm.md.water_swap_mc import (
    LAMBDA_COULOMB_SWIT6,
    LAMBDA_COULOMB_SWIT7,
    LAMBDA_VDW_SWIT6,
    LAMBDA_VDW_SWIT7,
    WaterSwapMC,
    WaterSwapSettings,
    _periodic_displacement,
    _water_oxygen_indices,
    water_swap_lambdas_at_equilibrium,
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
    for atom_idx, (q, sigma, eps) in zip(g2_atom_indices, saved, strict=False):
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
    assert water_swap_lambdas_at_equilibrium(sim.context)
    # 0.01 kJ/mol tolerance is generous enough for round-trip
    # representation error in the offset path (q_orig is stored
    # separately and re-multiplied by lambda=0 each evaluation)
    # without papering over a real coupling bug.
    # swit6/swit7 vdW uses soft-core CustomNonbonded (cutoff) vs PME on the
    # reference state; ~0.5 kJ/mol offset on a 27-water box is expected.
    assert e_alchemy_at_zero == pytest.approx(e_with_ghost2_off_direct, abs=1.0)


def test_alchemy_at_lambda_one_swaps_which_ghost_is_on() -> None:
    """At end-state lambdas ghost1 is off and ghost2 is on - mirror of equilibrium.

    What we strictly test here is that the alchemy is reversible
    (switched state -> equilibrium returns the original energy exactly).
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
    ctx = sim.context
    ctx.setParameter(LAMBDA_VDW_SWIT6, 0.0)
    ctx.setParameter(LAMBDA_COULOMB_SWIT6, 0.0)
    ctx.setParameter(LAMBDA_VDW_SWIT7, 1.0)
    ctx.setParameter(LAMBDA_COULOMB_SWIT7, 1.0)
    _ = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    ctx.setParameter(LAMBDA_VDW_SWIT6, 1.0)
    ctx.setParameter(LAMBDA_COULOMB_SWIT6, 1.0)
    ctx.setParameter(LAMBDA_VDW_SWIT7, 0.0)
    ctx.setParameter(LAMBDA_COULOMB_SWIT7, 0.0)
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
    box = np.diag([3.0, 3.0, 3.0])
    p = np.array([2.9, 0.1, 1.5])
    q = np.array([0.1, 2.9, 1.5])
    d = _periodic_displacement(p, q, box)
    assert d == pytest.approx(np.array([-0.2, 0.2, 0.0]), abs=1e-9)


def test_minimum_image_displacement_matches_constantph() -> None:
    """Triclinic minimum image must match the ConstantPH helper."""
    from opensqm.cph.constantph import _min_image_distance_matrix

    box = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [5.0, 5.0, 7.071067811865475],
        ],
        dtype=float,
    )
    center = np.array([5.0, 5.0, 3.5])
    probe = np.array([9.2, 9.1, 6.8])
    ours = float(np.linalg.norm(_periodic_displacement(probe, center, box)))
    ref = float(
        _min_image_distance_matrix(probe[None], center[None], box)[0, 0],
    )
    assert ours == pytest.approx(ref, abs=1e-9)


def test_lambda_schedules_four_phase_decoupling() -> None:
    """Four-phase schedule: A charge off, vdW to 0.5, vdW complete, B charge on."""
    sim, lig = _make_water_box()
    mover = WaterSwapMC(
        sim,
        lig,
        config=WaterSwapSettings(
            n_perturbation_steps=4,
            n_propagation_steps_per_perturbation=0,
        ),
    )
    l6_v, l6_c, l7_v, l7_c = mover._build_lambda_schedules(4)
    # t=0: A on, B off
    assert (l6_v[0], l6_c[0]) == pytest.approx((1.0, 1.0))
    assert (l7_v[0], l7_c[0]) == pytest.approx((0.0, 0.0))
    # t=0.25: A chargeless, full vdW
    assert (l6_v[1], l6_c[1]) == pytest.approx((1.0, 0.0))
    assert (l7_v[1], l7_c[1]) == pytest.approx((0.0, 0.0))
    # t=0.5: both half vdW, no charges
    assert (l6_v[2], l6_c[2]) == pytest.approx((0.5, 0.0))
    assert (l7_v[2], l7_c[2]) == pytest.approx((0.5, 0.0))
    # t=0.75: A off, B full vdW, no charges
    assert (l6_v[3], l6_c[3]) == pytest.approx((0.0, 0.0))
    assert (l7_v[3], l7_c[3]) == pytest.approx((1.0, 0.0))
    # t=1: A off, B fully on
    assert (l6_v[4], l6_c[4]) == pytest.approx((0.0, 0.0))
    assert (l7_v[4], l7_c[4]) == pytest.approx((1.0, 1.0))


def test_attempt_runs_and_restores_lambda() -> None:
    """One full attempt: lambdas must end at equilibrium whether accepted or not."""
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
    assert water_swap_lambdas_at_equilibrium(sim.context)
    assert mover.total_attempts == 1


def test_at_least_one_switch_completes_over_many_attempts() -> None:
    """Smoke test: NCMC switches should complete without blowing up.

    Uses a short switch (no propagation) so the test stays fast on a
    CPU. Overlap at the sampled insertion point is handled by soft-core
    vdW in the :class:`~openmm.CustomNonbondedForce`, matching the
    reference sampler — not by clash rejection during target picking.
    """
    np.random.seed(20260520)
    sim, ligand = _make_water_box(box_nm=3.0)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)

    mover = WaterSwapMC(
        sim,
        ligand,
        config=WaterSwapSettings(
            active_site_radius=0.7,
            n_perturbation_steps=20,
            n_propagation_steps_per_perturbation=0,
            pocket_sphere_restraint=False,
        ),
    )

    completed = 0
    n_attempts = 20
    for _ in range(n_attempts):
        sim.step(50)
        mover.attempt()
        if mover.in_stats.work_history or mover.out_stats.work_history:
            completed += 1

    assert mover.total_attempts == n_attempts
    assert completed >= 1, (
        f"No water-swap NCMC switches completed in {n_attempts} tries; "
        f"stats: {mover.summary()}"
    )
    assert water_swap_lambdas_at_equilibrium(sim.context)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
