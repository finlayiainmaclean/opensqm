"""Tests for CTMD: the c(t) bias offset, commit detection, and scoring."""

from __future__ import annotations

import math

import numpy as np
import openmm
import pytest
from openmm import app, unit
from openmm.app.metadynamics import BiasVariable, Metadynamics

from opensqm.ctmd.config import CTMDSettings
from opensqm.ctmd.metad import (
    CTMDTrajectory,
    alignment_atoms,
    ct,
    ct_from_bias,
    image_ligand_with_protein,
    ligand_rmsd_force,
    run_ctmd_replica,
)
from opensqm.ctmd.run_ctmd import collect_replicas
from opensqm.ctmd.score import bootstrap_score, commit_frame, rank_ligands, score_replica
from opensqm.md.align import kabsch_rt
from opensqm.modbind.states import PreparedState

KT_300 = (unit.MOLAR_GAS_CONSTANT_R * 300 * unit.kelvin).value_in_unit(unit.kilojoule_per_mole)
GAMMA = 10.0


def _soft_max(bias: np.ndarray, kt: float, gamma: float) -> float:
    """c(t) by the identity c = kT ln E_P[e^{beta V}], P proportional to e^{beta V/(gamma-1)}.

    An independent route to the same number, written naively without logsumexp.
    """
    weights = np.exp(bias / kt / (gamma - 1.0))
    return kt * math.log(float(np.sum(weights * np.exp(bias / kt)) / np.sum(weights)))


def test_uniform_bias_returns_its_own_height() -> None:
    # The only value c(t) can take when every grid point carries the same bias,
    # and the check that pins the gamma/(gamma-1) algebra.
    for height in (0.0, 3.5, 42.0):
        bias = np.full(64, height)
        assert ct_from_bias(bias, KT_300, GAMMA) == pytest.approx(height)


def test_matches_the_soft_max_identity() -> None:
    rng = np.random.default_rng(0)
    bias = rng.uniform(0.0, 40.0, size=200)
    assert ct_from_bias(bias, KT_300, GAMMA) == pytest.approx(_soft_max(bias, KT_300, GAMMA))


def test_lies_between_the_smallest_and_largest_bias() -> None:
    rng = np.random.default_rng(1)
    bias = rng.uniform(0.0, 60.0, size=200)
    value = ct_from_bias(bias, KT_300, GAMMA)
    assert bias.min() <= value <= bias.max()


def test_grid_spacing_cancels() -> None:
    # Refining the grid must not move c(t): the spacing divides out of the ratio.
    bias = np.linspace(0.0, 30.0, 100)
    assert ct_from_bias(bias, KT_300, GAMMA) == pytest.approx(
        ct_from_bias(np.repeat(bias, 3), KT_300, GAMMA)
    )


def test_survives_a_bias_deep_enough_to_overflow() -> None:
    # beta*V of 400 overflows a plain exp; logsumexp must carry it.
    bias = np.linspace(0.0, 1000.0, 200)
    assert np.isfinite(ct_from_bias(bias, KT_300, GAMMA))


def test_unbiased_metadynamics_has_zero_offset() -> None:
    # Exercises the getFreeEnergy() inversion against a real Metadynamics.
    system = openmm.System()
    system.addParticle(1.0 * unit.dalton)
    cv = openmm.CustomExternalForce("x")
    cv.addParticle(0, [])
    variable = BiasVariable(cv, 0.0, 1.0, 0.05, gridWidth=32)
    meta = Metadynamics(
        system, [variable], 300 * unit.kelvin, GAMMA, 1.0 * unit.kilojoule_per_mole, 10
    )
    assert ct(meta) == pytest.approx(0.0, abs=1e-9)


def test_commit_frame_is_the_start_of_the_first_sustained_excursion() -> None:
    rmsd = np.array([0.1, 0.1, 0.7, 0.7, 0.7, 0.7])
    assert commit_frame(rmsd, cutoff=0.6, commit_frames=4) == 2


def test_a_brief_excursion_does_not_commit() -> None:
    rmsd = np.array([0.1, 0.7, 0.7, 0.1, 0.7, 0.1])
    assert commit_frame(rmsd, cutoff=0.6, commit_frames=3) == -1
    assert commit_frame(np.full(10, 0.1), cutoff=0.6, commit_frames=3) == -1


def test_an_earlier_excursion_wins_over_a_longer_later_one() -> None:
    rmsd = np.array([0.7, 0.7, 0.1, 0.7, 0.7, 0.7, 0.7])
    assert commit_frame(rmsd, cutoff=0.6, commit_frames=2) == 0


def test_score_replica_reads_ct_when_the_ligand_left() -> None:
    config = CTMDSettings(commit_cutoff=0.6 * unit.nanometer, commit_time=0.4 * unit.picoseconds)
    assert config.commit_frames == 2
    trajectory = CTMDTrajectory(
        rmsd_nm=np.array([0.1, 0.2, 0.7, 0.7, 0.9]),
        ct_kj=np.array([0.0, 5.0, 11.0, 14.0, 20.0]),
        frame_interval_ps=0.2,
    )
    ct_value, residence_ns, committed = score_replica(trajectory, config)
    # Frame 2 is where it left, so 11.0 kJ/mol -- not the 20.0 it reached later.
    assert (ct_value, committed) == (11.0, True)
    assert residence_ns == pytest.approx(2 * 0.2 / 1000.0)


def test_a_replica_that_never_leaves_is_scored_at_its_last_frame() -> None:
    config = CTMDSettings(commit_cutoff=0.6 * unit.nanometer, commit_time=0.4 * unit.picoseconds)
    trajectory = CTMDTrajectory(
        rmsd_nm=np.array([0.1, 0.2, 0.3]),
        ct_kj=np.array([0.0, 5.0, 9.0]),
        frame_interval_ps=0.2,
    )
    assert score_replica(trajectory, config) == (9.0, pytest.approx(0.0004), False)


def test_bootstrap_takes_the_minimum_and_collapses_when_nothing_is_spare() -> None:
    spread = bootstrap_score([10.0, 20.0, 30.0], [1.0, 2.0, 3.0], n_select=2, repeats=500, seed=0)
    assert 10.0 <= spread["ct_score"] < 20.0
    assert spread["ct_score_std"] > 0.0

    # Drawing 3 of 3 without replacement always gives the same set.
    exact = bootstrap_score([10.0, 20.0, 30.0], [1.0, 2.0, 3.0], n_select=3, repeats=10, seed=0)
    assert exact["ct_score"] == pytest.approx(10.0)
    assert exact["ct_score_std"] == pytest.approx(0.0)


def test_bootstrap_refuses_to_score_too_few_replicas() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        bootstrap_score([10.0, 20.0], [1.0, 2.0], n_select=3, repeats=10, seed=0)


def test_ranking_breaks_ties_on_residence_time_and_sinks_failures() -> None:
    names = ["weak", "tied_brief", "tied_long", "failed"]
    ct_scores = [10.0, 30.0, 29.0, math.nan]
    residence_times = [5.0, 1.0, 4.0, 0.0]
    assert rank_ligands(names, ct_scores, residence_times, tie_tolerance=2.5) == [
        "tied_long",
        "tied_brief",
        "weak",
        "failed",
    ]


def test_a_gap_wider_than_the_tolerance_is_not_a_tie() -> None:
    # 30 and 26 differ by more than 1 kT, so residence time must not reorder them.
    assert rank_ligands(["a", "b"], [30.0, 26.0], [1.0, 9.0], tie_tolerance=2.5) == ["a", "b"]


def _toy_state(n_protein: int = 60, n_ligand: int = 4) -> PreparedState:
    """A toy complex: a rigid-ish 'protein' cloud plus a small 'ligand'."""
    rng = np.random.default_rng(0)
    system = openmm.System()
    topology = app.Topology()
    chain = topology.addChain()
    positions = np.vstack(
        [
            rng.normal(0.0, 1.0, (n_protein, 3)),
            rng.normal(0.0, 0.15, (n_ligand, 3)) + np.array([1.0, 0.0, 0.0]),
        ]
    )
    for resname, count in (("ALA", n_protein), ("LIG", n_ligand)):
        residue = topology.addResidue(resname, chain)
        for _ in range(count):
            system.addParticle(12.0 * unit.dalton)
            topology.addAtom("C", app.element.carbon, residue)
    return PreparedState(
        topology=topology,
        positions=positions * unit.nanometer,
        system=system,
        ligand_indices=list(range(n_protein, n_protein + n_ligand)),
        is_bound=True,
    )


def _read_cv(state: PreparedState, xyz_nm: np.ndarray) -> float:
    """Evaluate the CV expression itself, which is the force's energy."""
    system = openmm.System()
    for _ in range(state.topology.getNumAtoms()):
        system.addParticle(12.0 * unit.dalton)
    force = ligand_rmsd_force(state)
    force.setForceGroup(1)
    system.addForce(force)
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.001 * unit.picosecond),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(xyz_nm * unit.nanometer)
    energy = context.getState(getEnergy=True, groups={1}).getPotentialEnergy()
    return energy.value_in_unit(unit.kilojoule_per_mole)


def _kabsch_ligand_rmsd(state: PreparedState, xyz_nm: np.ndarray) -> float:
    """The same quantity by direct superposition, as the independent answer."""
    ref = np.asarray(state.positions.value_in_unit(unit.nanometer))
    protein = alignment_atoms(state)
    rotation, translation = kabsch_rt(xyz_nm[protein], ref[protein])
    moved = xyz_nm @ rotation.T + translation
    delta = moved[state.ligand_indices] - ref[state.ligand_indices]
    return float(np.sqrt((delta**2).sum(axis=1).mean()))


def test_the_cv_tracks_the_ligand_leaving_the_pocket() -> None:
    state = _toy_state()
    ref = np.asarray(state.positions.value_in_unit(unit.nanometer))
    for displacement in (0.05, 0.2, 0.6):
        moved = ref.copy()
        moved[state.ligand_indices] += np.array([displacement, 0.0, 0.0])
        assert _read_cv(state, moved) == pytest.approx(_kabsch_ligand_rmsd(state, moved), abs=0.005)
        assert _read_cv(state, moved) == pytest.approx(displacement, abs=0.005)


def test_the_cv_ignores_rigid_motion_of_the_whole_complex() -> None:
    # The regression test for the defect this replaced: a single RMSDForce over
    # the ligand superimposes what it measures, so it scored 0 for a ligand that
    # had left the pocket AND 0 here. Only the second is correct.
    state = _toy_state()
    ref = np.asarray(state.positions.value_in_unit(unit.nanometer))
    spin = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert _read_cv(state, ref) == pytest.approx(0.0, abs=1e-3)
    assert _read_cv(state, ref @ spin.T + np.array([5.0, 3.0, -2.0])) == pytest.approx(
        0.0, abs=1e-3
    )


def test_the_alignment_frame_must_outnumber_the_ligand() -> None:
    with pytest.raises(ValueError, match="must outnumber"):
        ligand_rmsd_force(_toy_state(n_protein=2, n_ligand=8))


def test_a_replica_biases_the_rmsd_and_accumulates_an_offset() -> None:
    # Smoke test for the force stack: RMSDForce -> BiasVariable -> Metadynamics.
    # One hill per frame, so c(t) must have moved off zero by the end.
    config = CTMDSettings(
        max_time=0.001 * unit.nanosecond,
        frame_interval=0.2 * unit.picoseconds,
        hill_interval=0.2 * unit.picoseconds,
        grid_bins=32,
    )
    state = _toy_state()
    trajectory = run_ctmd_replica(state, config, seed=7)

    assert len(trajectory.rmsd_nm) == config.max_frames + 1  # frame 0 is the unbiased start
    assert len(trajectory.ct_kj) == len(trajectory.rmsd_nm)
    assert trajectory.rmsd_nm[0] == pytest.approx(0.0, abs=1e-3)  # sqrt floor + float32
    assert trajectory.ct_kj[0] == pytest.approx(0.0, abs=1e-9)
    assert trajectory.ct_kj[-1] > 0.0
    # The bias belongs to the copy, never to the cached prepared state.
    assert state.system is not None
    assert state.system.getNumForces() == 0


def test_a_run_too_small_to_score_is_rejected_before_it_starts(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be at least"):
        collect_replicas(
            _toy_state(),
            CTMDSettings(n_replicas=2, n_score_replicas=3),
            checkpoint_dir=tmp_path / "checkpoints",
            trajectory_dir=tmp_path / "trajectories",
        )


def test_the_imaging_bond_adds_no_energy() -> None:
    # It exists only so OpenMM treats ligand and protein as one molecule for
    # periodic imaging; if it ever contributed a force it would perturb the run.
    state = _toy_state()
    system = openmm.System()
    for _ in range(state.topology.getNumAtoms()):
        system.addParticle(12.0 * unit.dalton)
    image_ligand_with_protein(system, state).setForceGroup(2)
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.001 * unit.picosecond),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(state.positions)
    result = context.getState(getEnergy=True, getForces=True, groups={2})
    assert result.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) == 0.0
    assert np.abs(result.getForces(asNumpy=True)).max() == 0.0
