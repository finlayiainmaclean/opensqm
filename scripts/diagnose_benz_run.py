#!/usr/bin/env python3
"""Diagnose a ModBinddG benzamidine-trypsin run against saved checkpoints.

Focuses on whether the *bound well* (0-2 A COM) is actually sampled long
enough for the population-reweighting to be meaningful.
"""

from pathlib import Path

import numpy as np

from opensqm.modbind.reweight import (
    R_KCAL_PER_MOL_K,
    _bin_ids,
    frame_weights,
    reweight_exponent,
    state_population,
    trajectory_state_population,
)

RUN = Path("runs/benz")
TEMP_K = 1200.0
FRAME_PS = 10.0
BIN = 4.0
BOUND_CUTOFF = 2.0


def load_bound() -> list[np.ndarray]:
    paths = sorted((RUN / "checkpoints").glob("bound_*.npy"))
    return [np.load(p) for p in paths]


def main() -> None:
    trajs = load_bound()
    exp = reweight_exponent(TEMP_K)
    print(f"T*={TEMP_K} K  ->  reweight exponent 1/lambda = T*/T = {exp:.3f}\n")

    print("=== Per-trajectory bound-well sampling ===")
    print(
        f"{'rep':>3} {'frames':>7} {'in<2A':>6} {'well_ns':>8} "
        f"{'maxbin_n':>8} {'reweighted_bound':>16}"
    )
    for i, t in enumerate(trajs):
        r = np.linalg.norm(t, axis=1)
        n_in = int(np.sum(r < BOUND_CUTOFF))
        # deepest occupied bin among bound frames
        bin_ids = _bin_ids(t, BIN)
        _, _inv, counts = np.unique(bin_ids, axis=0, return_inverse=True, return_counts=True)
        maxbin = int(counts.max())
        w = frame_weights(t, temperature=TEMP_K, bin_size=BIN)
        bound_w = state_population(t, w, BOUND_CUTOFF)
        well_ns = n_in * FRAME_PS * 1e-3
        print(f"{i:>3} {len(t):>7} {n_in:>6} {well_ns:>8.3f} {maxbin:>8} {bound_w:>16.1f}")

    bound_pop = trajectory_state_population(
        trajs, temperature=TEMP_K, bin_size=BIN, cutoff=BOUND_CUTOFF
    )
    print(f"\nmean reweighted bound population / replica: {bound_pop:.1f}")

    # What bound population is needed for the paper answer?
    rt = R_KCAL_PER_MOL_K * 300.0
    vol_corr = 0.688
    p_unbound = 404.9  # from results.csv (explicit, time-normalised)
    for target in (-5.57,):
        target_comp = target - vol_corr
        need_ratio = np.exp(target_comp / rt)  # P_u / P_b
        need_pb = p_unbound / need_ratio
        print(f"\nfor dG = {target}: need P_unbound/P_bound = {need_ratio:.2e}")
        print(f"  with explicit P_unbound={p_unbound:.0f} -> need P_bound ~ {need_pb:.3e}")
        # frames in deepest bin to reach that, per-traj averaged
        n_needed = need_pb ** (1.0 / exp)
        print(
            f"  i.e. deepest bound bin needs ~{n_needed:.0f} frames "
            f"(~{n_needed * FRAME_PS * 1e-3:.2f} ns in the well per replica)"
        )

    print("\n=== sanity: recompute ΔG_comp from the two populations ===")
    dg_comp = rt * np.log(p_unbound / bound_pop)
    print(f"RT ln(P_u/P_b) = {dg_comp:.2f} kcal/mol   (results.csv: -2.71)")


if __name__ == "__main__":
    main()
