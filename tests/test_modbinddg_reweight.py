"""Tests for ModBinddG population reweighting and free-energy estimation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from openmm import unit

from opensqm.modbind.config import ModBindDGSettings
from opensqm.modbind.reweight import (
    compute_delta_g,
    reweight_state,
    rt_kcal,
)


def _single_bin_trajectory(n_frames: int, bin_coord: tuple[int, int, int]) -> np.ndarray:
    return np.full((n_frames, 3), bin_coord, dtype=np.float64)


def test_uniform_temperature_matches_single_exponent() -> None:
    trajectories = [_single_bin_trajectory(3, (0, 0, 0)), _single_bin_trajectory(5, (0, 0, 0))]
    single = reweight_state(trajectories, bin_size=4.0, exponent=2.0, radius=10.0)
    per_replica = reweight_state(trajectories, bin_size=4.0, exponent=(2.0, 2.0), radius=10.0)
    assert single.total == pytest.approx(per_replica.total)
    np.testing.assert_allclose(single.per_replica, per_replica.per_replica)


def test_different_replica_temperatures_change_population() -> None:
    trajectories = [_single_bin_trajectory(4, (0, 0, 0)), _single_bin_trajectory(4, (0, 0, 0))]
    low_t = reweight_state(trajectories, bin_size=4.0, exponent=(1.5, 1.5), radius=10.0)
    mixed = reweight_state(trajectories, bin_size=4.0, exponent=(1.5, 2.5), radius=10.0)
    assert mixed.total != pytest.approx(low_t.total)
    assert mixed.per_replica[0] == pytest.approx(low_t.per_replica[0])
    assert mixed.per_replica[1] > low_t.per_replica[1]


def test_unbound_reweighted_with_its_own_exponent() -> None:
    # Bound hot (900 K) and unbound at 300 K carry DIFFERENT exponents. The
    # unbound exponent is T_u / T_ref = 1, so its per-escape population scales
    # linearly with frames (NOT as frames ** bound_exponent).
    bound = [_single_bin_trajectory(40, (0, 0, 0))]
    cfg = ModBindDGSettings(
        bound_temperature=900 * unit.kelvin,
        unbound_temperature=300 * unit.kelvin,
        n_replicas=1,
        unbound_frame_interval=10.0 * unit.picoseconds,
    )
    assert cfg.reweight_exponent == pytest.approx(3.0)
    assert cfg.unbound_reweight_exponent == pytest.approx(1.0)
    rt = rt_kcal(300.0)
    u40 = compute_delta_g(bound, [_single_bin_trajectory(40, (1, 0, 0))], config=cfg, rt=rt)
    u80 = compute_delta_g(bound, [_single_bin_trajectory(80, (1, 0, 0))], config=cfg, rt=rt)
    assert u80["unbound_population"] / u40["unbound_population"] == pytest.approx(
        2.0**cfg.unbound_reweight_exponent
    )


def test_frame_rate_factor_scales_with_unbound_exponent() -> None:
    # The capture-interval correction uses the UNBOUND state's own exponent, since
    # it re-expresses the unbound counts at the bound (reference) frame interval:
    # (unbound_dt / bound_dt) ** unbound_exponent.
    occ_b = [_single_bin_trajectory(40, (0, 0, 0))]
    occ_u = [_single_bin_trajectory(40, (1, 0, 0))]
    base = ModBindDGSettings(
        bound_temperature=900 * unit.kelvin,
        unbound_temperature=300 * unit.kelvin,
        n_replicas=1,
        unbound_frame_interval=10.0 * unit.picoseconds,  # equal -> factor 1
    )
    fine = base.copy(update={"unbound_frame_interval": 1.0 * unit.picoseconds})
    rt = rt_kcal(300.0)
    pop_base = compute_delta_g(occ_b, occ_u, config=base, rt=rt)["unbound_population"]
    pop_fine = compute_delta_g(occ_b, occ_u, config=fine, rt=rt)["unbound_population"]
    assert pop_fine / pop_base == pytest.approx((1.0 / 10.0) ** base.unbound_reweight_exponent)


def test_delta_g_invariant_to_unequal_escape_counts() -> None:
    # The user's scenario: bound at 900 K (exponent 3) with N_b replicas, unbound
    # at 300 K (exponent 1) with N_u replicas, N_b != N_u. Per-escape
    # normalisation must make dG invariant to N_b and N_u *independently*, so the
    # 8-vs-32 inequality introduces no bias even though the exponents differ.
    bound_one = [_single_bin_trajectory(40, (0, 0, 0))]
    unbound_one = [_single_bin_trajectory(60, (1, 0, 0))]
    rt = rt_kcal(300.0)

    def dg(n_bound: int, n_unbound: int) -> float:
        cfg = ModBindDGSettings(
            bound_temperature=900 * unit.kelvin,
            unbound_temperature=300 * unit.kelvin,
            n_replicas=n_bound,
        )
        return compute_delta_g(bound_one * n_bound, unbound_one * n_unbound, config=cfg, rt=rt)[
            "delta_g"
        ]

    reference = dg(1, 1)
    assert math.isfinite(reference)
    # Vary bound and unbound escape counts independently, including 8 vs 32.
    for n_bound, n_unbound in [(8, 8), (1, 32), (8, 32), (32, 8), (16, 4)]:
        assert dg(n_bound, n_unbound) == pytest.approx(reference), (
            f"dG drifted at N_bound={n_bound}, N_unbound={n_unbound}"
        )


def test_delta_g_invariant_to_replica_count_explicit() -> None:
    # Duplicating the (identical) escape set must not change dG: reweighting the
    # per-escape mean occupancy makes each state intensive, so N copies of the
    # same trajectory give the same answer as one. Raw pooled-count reweighting
    # drifted by -RT (1/lambda - 1) ln(N) instead (verified in the toy model).
    bound_one = [_single_bin_trajectory(40, (0, 0, 0))]
    unbound_one = [_single_bin_trajectory(40, (1, 0, 0))]
    config = ModBindDGSettings(bound_temperature=900 * unit.kelvin, n_replicas=1)
    rt = rt_kcal(300.0)
    dg1 = compute_delta_g(bound_one, unbound_one, config=config, rt=rt)["delta_g"]
    assert math.isfinite(dg1)
    for n in (2, 4, 8, 16):
        cfg_n = config.copy(update={"n_replicas": n})
        dg_n = compute_delta_g(bound_one * n, unbound_one * n, config=cfg_n, rt=rt)["delta_g"]
        assert dg_n == pytest.approx(dg1), f"dG drifted at N={n}: {dg_n} vs {dg1}"
