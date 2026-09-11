"""Tests for CTMD: the c(t) bias offset, commit detection, and scoring."""

from __future__ import annotations

import math

import numpy as np
import openmm
import pytest
from openmm import app, unit
from openmm.app.metadynamics import BiasVariable, Metadynamics

from opensqm.ctmd.config import CTMDSettings
from opensqm.ctmd.metad import CTMDTrajectory, ct, ct_from_bias, run_ctmd_replica
from opensqm.ctmd.run_ctmd import collect_replicas
from opensqm.ctmd.score import bootstrap_score, commit_frame, rank_ligands, score_replica
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


def _toy_state() -> PreparedState:
    """A three-particle 'ligand' in vacuum, enough to drive the real force stack."""
    system = openmm.System()
    topology = app.Topology()
    residue = topology.addResidue("LIG", topology.addChain())
    for _ in range(3):
        system.addParticle(12.0 * unit.dalton)
        topology.addAtom("C", app.element.carbon, residue)
    positions = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.0, 0.15, 0.0]]) * unit.nanometer
    return PreparedState(
        topology=topology,
        positions=positions,
        system=system,
        ligand_indices=[0, 1, 2],
        is_bound=True,
    )


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
    assert trajectory.rmsd_nm[0] == pytest.approx(0.0, abs=1e-4)  # single-precision RMSD
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
