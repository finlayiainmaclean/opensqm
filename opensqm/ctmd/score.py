"""Turn CTMD replica trajectories into a triage score and a ligand ranking.

A replica is scored at the frame where the ligand first commits to leaving: the
start of the first run of consecutive frames above the RMSD cutoff that lasts a
full commit window. c(t) there is the reversible work the bias had to supply to
get the ligand out, so a larger value means a better binder. A replica that
never commits is scored at its last frame, which underestimates it -- the run
was capped, not finished.

The ligand score is the minimum c(t) over a subset of replicas, averaged over
bootstrap draws of that subset from the replicas actually collected. Taking the
minimum is deliberate: one replica finding a cheap exit is evidence the pose is
weak, however well the others held.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from openmm import unit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opensqm.ctmd.config import CTMDSettings
    from opensqm.ctmd.metad import CTMDTrajectory


def commit_frame(rmsd_nm: np.ndarray, *, cutoff: float, commit_frames: int) -> int:
    """First frame of the first sustained excursion above ``cutoff``, else -1.

    ``commit_frames`` consecutive frames must all exceed the cutoff. Returns the
    index of the first of them, not the last, so the score is read when the
    ligand left rather than a commit window later.
    """
    above = np.asarray(rmsd_nm) > cutoff
    if above.size < commit_frames:
        return -1
    runs = np.convolve(above, np.ones(commit_frames), mode="valid")
    start = int(np.argmax(runs))
    return start if runs[start] == commit_frames else -1


def score_replica(trajectory: CTMDTrajectory, config: CTMDSettings) -> tuple[float, float, bool]:
    """Return ``(c(t) in kJ/mol, residence time in ns, committed)`` for one replica."""
    index = commit_frame(
        trajectory.rmsd_nm,
        cutoff=config.commit_cutoff.value_in_unit(unit.nanometer),
        commit_frames=config.commit_frames,
    )
    committed = index >= 0
    if not committed:
        index = len(trajectory.rmsd_nm) - 1
    residence_ns = index * trajectory.frame_interval_ps / 1000.0
    return float(trajectory.ct_kj[index]), residence_ns, committed


def bootstrap_score(
    ct_values: Sequence[float],
    residence_times: Sequence[float],
    *,
    n_select: int,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    """Mean and spread of the minimum over ``n_select`` replicas drawn without replacement.

    With exactly ``n_select`` replicas collected every draw is the same set, so
    the spread is zero and the mean is the plain minimum.
    """
    cts = np.asarray(ct_values, dtype=np.float64)
    rts = np.asarray(residence_times, dtype=np.float64)
    if cts.size < n_select:
        raise ValueError(f"Need at least {n_select} replicas to score, got {cts.size}")

    rng = np.random.default_rng(seed)
    draws = np.stack([rng.choice(cts.size, n_select, replace=False) for _ in range(repeats)])
    ct_min = cts[draws].min(axis=1)
    rt_min = rts[draws].min(axis=1)
    return {
        "ct_score": float(ct_min.mean()),
        "ct_score_std": float(ct_min.std()),
        "residence_time_ns": float(rt_min.mean()),
        "residence_time_std": float(rt_min.std()),
    }


def rank_ligands(
    names: Sequence[str],
    ct_scores: Sequence[float],
    residence_times: Sequence[float],
    *,
    tie_tolerance: float = 2.5,
) -> list[str]:
    """Order ligands best-first for triage.

    Sorts by descending c(t), then re-orders each run of ligands within
    ``tie_tolerance`` kJ/mol of the run's leader by descending residence time.
    Ligands whose run failed (NaN c(t)) sort last. The grouping anchors on the
    leader rather than on a running minimum, so a long chain of small gaps
    cannot slide a weak ligand into a strong group.
    """

    def key(index: int) -> float:
        value = ct_scores[index]
        return -math.inf if math.isnan(value) else value

    ranked: list[int] = []
    group: list[int] = []
    for index in sorted(range(len(names)), key=lambda i: -key(i)):
        if group and key(group[0]) - key(index) > tie_tolerance:
            ranked.extend(sorted(group, key=lambda i: -residence_times[i]))
            group = []
        group.append(index)
    ranked.extend(sorted(group, key=lambda i: -residence_times[i]))
    return [names[i] for i in ranked]
