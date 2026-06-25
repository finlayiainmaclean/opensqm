"""Tests for ModBinddG per-replica temperature reweighting."""

from __future__ import annotations

import math

import numpy as np
import pytest
from openmm import unit

from opensqm.modbinddg.config import ModBindDGSettings
from opensqm.modbinddg.reweight import (
    compute_delta_g,
    einstein_smoluchowski_unbound,
    predict_escape_temperature_calibrated,
    reweight_state,
    rt_kcal,
)


def _single_bin_trajectory(n_frames: int, bin_coord: tuple[int, int, int]) -> np.ndarray:
    return np.full((n_frames, 3), bin_coord, dtype=np.float64)


def test_uniform_temperature_matches_single_exponent() -> None:
    trajectories = [_single_bin_trajectory(3, (0, 0, 0)), _single_bin_trajectory(5, (0, 0, 0))]
    single = reweight_state(
        trajectories, bin_size=4.0, exponent=2.0, radius=10.0
    )
    per_replica = reweight_state(
        trajectories, bin_size=4.0, exponent=(2.0, 2.0), radius=10.0
    )
    assert single.total == pytest.approx(per_replica.total)
    np.testing.assert_allclose(single.per_replica, per_replica.per_replica)


def test_different_replica_temperatures_change_population() -> None:
    trajectories = [_single_bin_trajectory(4, (0, 0, 0)), _single_bin_trajectory(4, (0, 0, 0))]
    low_t = reweight_state(
        trajectories, bin_size=4.0, exponent=(1.5, 1.5), radius=10.0
    )
    mixed = reweight_state(
        trajectories, bin_size=4.0, exponent=(1.5, 2.5), radius=10.0
    )
    assert mixed.total != pytest.approx(low_t.total)
    assert mixed.per_replica[0] == pytest.approx(low_t.per_replica[0])
    assert mixed.per_replica[1] > low_t.per_replica[1]


def test_compute_delta_g_accepts_per_replica_config() -> None:
    bound = [_single_bin_trajectory(4, (0, 0, 0))]
    unbound = [_single_bin_trajectory(4, (10, 10, 10))]
    uniform = ModBindDGSettings(
        temperature=650 * unit.kelvin,
        n_replicas=1,
        unbound_mode="explicit",
    )
    explicit = ModBindDGSettings(
        temperature=650 * unit.kelvin,
        replica_temperatures=(650.0,),
        n_replicas=1,
        unbound_mode="explicit",
    )
    rt = rt_kcal(300.0)
    uniform_result = compute_delta_g(bound, unbound, config=uniform, rt=rt)
    explicit_result = compute_delta_g(bound, unbound, config=explicit, rt=rt)
    assert uniform_result["delta_g"] == pytest.approx(explicit_result["delta_g"])


def test_einstein_unbound_uses_completed_bound_replica_count() -> None:
    config = ModBindDGSettings(
        temperature=650 * unit.kelvin,
        n_replicas=8,
        unbound_mode="einstein",
    )
    rt = rt_kcal(300.0)
    pop5, _ = einstein_smoluchowski_unbound(config, rt=rt, n_bound_escapes=5)
    pop8, _ = einstein_smoluchowski_unbound(config, rt=rt, n_bound_escapes=8)
    assert pop5 / pop8 == pytest.approx(5 / 8)

    bound = [_single_bin_trajectory(4, (0, 0, 0))] * 5
    partial_config = config.copy(update={"n_replicas": 5})
    result = compute_delta_g(bound, [], config=partial_config, rt=rt)
    assert result["n_bound_escapes"] == 5
    assert result["unbound_population"] == pytest.approx(pop5)



def test_calibrated_escape_temperature_matches_650k_anchor() -> None:
    # |ΔG°|≈5.5 kcal/mol ligands escape in ~1–4 ns at 650 K.
    predicted = predict_escape_temperature_calibrated(
        temperature_K=650.0,
        escape_time_ns=2.0,
        binding_dg_kcal=-5.5,
        target_escape_time_ns=1.0,
    )
    assert 650.0 < predicted < 750.0


def test_calibrated_escape_temperature_weaker_binding_needs_cooler() -> None:
    strong = predict_escape_temperature_calibrated(
        temperature_K=900.0,
        escape_time_ns=0.2,
        binding_dg_kcal=-5.5,
        target_escape_time_ns=1.0,
    )
    weak = predict_escape_temperature_calibrated(
        temperature_K=900.0,
        escape_time_ns=0.2,
        binding_dg_kcal=-2.5,
        target_escape_time_ns=1.0,
    )
    assert weak < strong
    assert 600.0 < weak < 850.0


def test_calibrated_escape_temperature_fast_high_t_extrapolation() -> None:
    predicted = predict_escape_temperature_calibrated(
        temperature_K=1200.0,
        escape_time_ns=0.07,
        binding_dg_kcal=-5.5,
        target_escape_time_ns=1.0,
    )
    assert 850.0 < predicted < 1000.0


def test_calibrated_escape_temperature_skips_none_or_zero_target() -> None:
    for target in (None, 0.0):
        predicted = predict_escape_temperature_calibrated(
            temperature_K=650.0,
            escape_time_ns=2.0,
            binding_dg_kcal=-5.5,
            target_escape_time_ns=target,
        )
        assert math.isnan(predicted)
