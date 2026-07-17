"""Population-based reweighting and free-energy estimation for ModBinddG.

Implements the discrete form of Eq. 14 of Sinko et al. (*PNAS* 2026):

    dG = RT ln( sum_unbound (p*)^(1/lambda) / sum_bound (p*)^(1/lambda) )
         - RT ln(Vu / V0)

Configurations are binned into equal-volume Cartesian cubes, bin counts are
raised to ``1/lambda`` (``lambda = T_ref / T_sim``) to reweight the
high-temperature populations back to the reference temperature, and each frame
is assigned an equal share of its bin's reweighted population. Bin counts are
pooled across replicas, but the reweighting exponent is applied per replica so
replicas simulated at different temperatures are normalised independently.

The pooled bin count is divided by the number of escapes *before* reweighting,
i.e. the quantity that is reweighted is the per-escape mean occupancy, not the
extensive pooled count. The bound and unbound states are reweighted with their
OWN exponents (``T_state / T_ref``), because the paper runs them at different
temperatures: a hot bound (unbinding) simulation and a 300 K unbound
(ligand-in-solvent) simulation whose exponent is therefore 1 (no reweighting).
Per-escape normalisation makes each state's population invariant to its escape
count for any exponent, so the Eq. 14 ratio is unaffected when the two states
are run for a *different* number of escapes (e.g. 8 bound vs 32 unbound) -- which
the paper permits: "it is also acceptable to normalize the populations of each
state if different numbers of escape simulations are performed for either
state." It gives the replica-count invariance the method assumes (Sinko et al.,
assumption 4); reweighting the raw extensive count instead would make ``dG``
drift by ``-RT (exponent - 1) ln(N)`` whenever the escape counts are unequal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from openmm import unit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opensqm.modbind.config import ModBindDGSettings

# Gas constant in kcal / (mol K).
R_KCAL_PER_MOL_K = 1.987204259e-3


def rt_kcal(temperature_k: float) -> float:
    """Return RT in kcal/mol at the given temperature."""
    return R_KCAL_PER_MOL_K * temperature_k


def estimate_delta_g_well(result: dict, *, rt: float) -> float:
    """Estimate the local binding-well depth from a running ModBinddG result.

    Uses ``ΔG° ≈ ΔG°_well + ΔG°_unbound`` with
    ``ΔG°_unbound = -RT ln(P_unbound)``, giving
    ``ΔG°_well = ΔG° + RT ln(P_unbound)``.
    """
    unbound_population = float(result["unbound_population"])
    if unbound_population <= 0 or not math.isfinite(result["delta_g"]):
        return math.nan
    return float(result["delta_g"] + rt * math.log(unbound_population))


@dataclass
class StatePopulation:
    """Reweighted population of a state plus per-replica diagnostics."""

    total: float
    per_replica: np.ndarray

    @property
    def effective_sample_size(self) -> float:
        """Kish effective sample size implied by the per-replica populations."""
        if self.per_replica.size == 0 or self.total <= 0:
            return 0.0
        return float(self.total**2 / np.sum(self.per_replica**2))

    @property
    def max_replica_fraction(self) -> float:
        """Fraction of the total population contributed by the largest replica."""
        if self.per_replica.size == 0 or self.total <= 0:
            return 0.0
        return float(np.max(self.per_replica) / self.total)


def pooled_bin_counts(
    coords_list: list[np.ndarray], *, bin_size: float
) -> dict[tuple[int, int, int], int]:
    """Pool all frames of all replicas into one cubic histogram of counts.

    Uses the same ``floor(coord / bin_size)`` binning as
    :func:`reweight_state` so per-frame weights derived from these counts are
    consistent with the population sums that enter Eq. 14.
    """
    counts: dict[tuple[int, int, int], int] = {}
    for coords in coords_list:
        bin_idx = np.floor(np.asarray(coords, dtype=np.float64) / bin_size).astype(int)
        for row in bin_idx:
            key = (int(row[0]), int(row[1]), int(row[2]))
            counts[key] = counts.get(key, 0) + 1
    return counts


def frame_weights(
    coords: np.ndarray,
    counts: dict[tuple[int, int, int], int],
    *,
    bin_size: float,
    exponent: float,
) -> np.ndarray:
    """Per-frame reweighted statistical weight ``count(bin)**(exponent-1)``.

    This is each frame's equal share of its bin's reweighted population
    (``count**exponent / count``), i.e. the weight implied by
    :func:`reweight_state`. ``counts`` must come from :func:`pooled_bin_counts`
    over the full trajectories so the normalisation matches the population sum.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 1:
        coords = coords[None, :]
    bin_idx = np.floor(coords / bin_size).astype(int)
    weights = np.empty(len(coords), dtype=np.float64)
    for i, row in enumerate(bin_idx):
        key = (int(row[0]), int(row[1]), int(row[2]))
        weights[i] = float(counts.get(key, 1)) ** (exponent - 1.0)
    return weights


def _replica_exponents(exponent: float | Sequence[float], n_replicas: int) -> tuple[float, ...]:
    if isinstance(exponent, (int, float)):
        return (float(exponent),) * n_replicas
    exponents = tuple(float(x) for x in exponent)
    if len(exponents) != n_replicas:
        raise ValueError(
            f"Expected {n_replicas} exponents for {n_replicas} replicas, got {len(exponents)}"
        )
    return exponents


def reweight_state(
    coords_list: list[np.ndarray],
    *,
    bin_size: float,
    exponent: float | Sequence[float],
    radius: float,
) -> StatePopulation:
    """Reweight pooled trajectories and sum populations within ``radius``.

    ``coords_list`` is a list of ``(n_frames, 3)`` COM-displacement arrays
    (Angstrom). All frames are pooled into a single cubic histogram; the
    per-escape mean bin count (pooled count / number of escapes) is raised to a
    per-replica ``exponent`` and shared equally among the frames in each bin.
    """
    if not coords_list:
        return StatePopulation(total=0.0, per_replica=np.empty(0))

    exponents = _replica_exponents(exponent, len(coords_list))

    counts: dict[tuple[int, int, int], int] = {}
    keys_per_replica: list[list[tuple[int, int, int]]] = []
    for coords in coords_list:
        bin_idx = np.floor(np.asarray(coords, dtype=np.float64) / bin_size).astype(int)
        keys = [tuple(int(c) for c in row) for row in bin_idx]
        keys_per_replica.append(keys)
        for key in keys:
            counts[key] = counts.get(key, 0) + 1

    n_replicas = len(coords_list)
    per_replica = np.zeros(n_replicas, dtype=np.float64)
    for replica_i, (coords, keys, exp) in enumerate(
        zip(coords_list, keys_per_replica, exponents, strict=False)
    ):
        radii = np.linalg.norm(np.asarray(coords, dtype=np.float64), axis=1)
        population = 0.0
        for r, key in zip(radii, keys, strict=False):
            if r <= radius:
                count = counts[key]
                # Reweight the per-escape *mean* occupancy (pooled count divided
                # by the number of escapes), not the extensive pooled count, so
                # the population is intensive and dG is invariant to the replica
                # count. ``/ count`` shares the bin population over its frames.
                mean_count = count / n_replicas
                population += mean_count**exp / count
        per_replica[replica_i] = population

    return StatePopulation(total=float(per_replica.sum()), per_replica=per_replica)


def radial_pmf(
    coords_list: list[np.ndarray],
    *,
    bin_size: float,
    exponent: float | Sequence[float],
    max_radius: float,
    rt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radial PMF (kcal/mol) vs COM distance, min normalised to 0.

    Uses the same cubic-bin reweighting as :func:`reweight_state`, then
    aggregates per-frame weights into radial shells of width ``bin_size``.
    """
    if not coords_list:
        return np.empty(0), np.empty(0)

    exponents = _replica_exponents(exponent, len(coords_list))

    counts: dict[tuple[int, int, int], int] = {}
    keys_per_replica: list[list[tuple[int, int, int]]] = []
    radii_per_replica: list[np.ndarray] = []
    for coords in coords_list:
        coords_arr = np.asarray(coords, dtype=np.float64)
        bin_idx = np.floor(coords_arr / bin_size).astype(int)
        keys = [tuple(int(c) for c in row) for row in bin_idx]
        keys_per_replica.append(keys)
        radii_per_replica.append(np.linalg.norm(coords_arr, axis=1))
        for key in keys:
            counts[key] = counts.get(key, 0) + 1

    n_replicas = len(coords_list)
    n_shells = max(1, math.ceil(max_radius / bin_size))
    shell_pop = np.zeros(n_shells, dtype=np.float64)
    for keys, radii, exp in zip(keys_per_replica, radii_per_replica, exponents, strict=False):
        for r, key in zip(radii, keys, strict=False):
            shell = int(r // bin_size)
            if 0 <= shell < n_shells:
                count = counts[key]
                # Per-escape mean occupancy, matching :func:`reweight_state`. The
                # PMF is min-normalised below, so this only removes a constant
                # offset, but keeps the reweighting identical across functions.
                mean_count = count / n_replicas
                shell_pop[shell] += mean_count**exp / count

    centers = (np.arange(n_shells) + 0.5) * bin_size
    with np.errstate(divide="ignore"):
        pmf = -rt * np.log(shell_pop)
    finite = np.isfinite(pmf)
    if finite.any():
        pmf[finite] -= pmf[finite].min()
    return centers, pmf


def bound_well_diagnostics(
    coords_list: list[np.ndarray],
    *,
    bin_size: float,
    boundary_radius: float,
    exponent: float,
    rt: float,
) -> dict:
    """Diagnose the sampled bound-well depth from pooled bound trajectories.

    Bins all bound frames into the same cubic histogram used by
    :func:`reweight_state` and reports, purely as diagnostics:

    * ``c_min`` -- occupancy of the most-populated bin (the well bottom).
    * ``c_boundary`` -- occupancy of the most-populated bin whose frames sit
      within one bin width of the absorbing boundary (the free-ligand reference);
      falls back to the outermost occupied bin.
    * ``c_min_radius`` -- mean COM radius (Angstrom) of the well-bottom bin.
    * ``delta_g_well`` -- the well depth this sampling can support,
      ``-exponent * RT * ln(c_min / c_boundary)``. This is boundary-anchored so it
      is independent of replica count and frame rate, but bounded in magnitude by
      ``exponent * RT * ln(frames_per_escape)``: if it is far shallower than the
      expected affinity, the bound trajectory is capture-limited (the boundary
      transit is faster than one ``bound_frame_interval``, so ``c_boundary`` is
      floored at ~1 frame/escape).
    """
    counts: dict[tuple[int, int, int], int] = {}
    radius_sum: dict[tuple[int, int, int], float] = {}
    for coords in coords_list:
        arr = np.asarray(coords, dtype=np.float64)
        radii = np.linalg.norm(arr, axis=1)
        bin_idx = np.floor(arr / bin_size).astype(int)
        for row, r in zip(bin_idx, radii, strict=False):
            key = (int(row[0]), int(row[1]), int(row[2]))
            counts[key] = counts.get(key, 0) + 1
            radius_sum[key] = radius_sum.get(key, 0.0) + float(r)

    if not counts:
        return {"c_min": 0, "c_boundary": 0, "c_min_radius": math.nan, "delta_g_well": math.nan}

    mean_radius = {k: radius_sum[k] / counts[k] for k in counts}

    # Well bottom: the most-occupied bin.
    c_min_key = max(counts, key=lambda k: counts[k])
    c_min = counts[c_min_key]

    # Boundary reference: most-occupied bin within one bin width of the boundary,
    # else the outermost occupied bin.
    boundary_bins = [
        c for k, c in counts.items() if abs(mean_radius[k] - boundary_radius) <= bin_size
    ]
    if boundary_bins:
        c_boundary = max(boundary_bins)
    else:
        c_boundary = counts[max(counts, key=lambda k: mean_radius[k])]

    delta_g_well = (
        -exponent * rt * math.log(c_min / c_boundary) if c_min > 0 and c_boundary > 0 else math.nan
    )

    return {
        "c_min": int(c_min),
        "c_boundary": int(c_boundary),
        "c_min_radius": float(mean_radius[c_min_key]),
        "delta_g_well": float(delta_g_well),
    }


def _compute_delta_g_core(
    bound_coords: list[np.ndarray],
    unbound_coords: list[np.ndarray],
    *,
    config: ModBindDGSettings,
    rt: float,
) -> dict:
    """Compute pooled dG and intermediate terms from reweighted populations."""
    exponent = config.reweight_exponent
    bound = reweight_state(
        bound_coords,
        bin_size=config.bin_size,
        exponent=exponent,
        radius=config.bound_state_radius,
    )

    n_bound_escapes = len(bound_coords)
    # The bound and unbound states are run at DIFFERENT temperatures, so each is
    # reweighted to T_ref with its OWN exponent T_state/T_ref (Eq. 4): the hot
    # bound state uses ``exponent`` (e.g. 900/300 = 3), while the unbound state,
    # run at 300 K, uses ``unbound_exponent`` = 1 (already canonical -> no
    # reweighting). This is the per-state generalisation of Eq. 14.
    #
    # The two "inequalities" between the states are handled independently:
    #   * unequal escape counts (e.g. 8 bound vs 32 unbound): ``reweight_state``
    #     divides each state's pooled bin count by its OWN number of escapes
    #     BEFORE raising to the exponent, so each state's population is intensive
    #     (invariant to its escape count) for ANY exponent. The ratio is thus
    #     unaffected by 8 != 32. The paper permits this: "it is also acceptable
    #     to normalize the populations of each state if different numbers of
    #     escape simulations are performed".
    #   * unequal frame-capture intervals: handled by ``frame_rate_factor`` below.
    unbound_exponent = config.unbound_reweight_exponent
    unbound = reweight_state(
        unbound_coords,
        bin_size=config.bin_size,
        exponent=unbound_exponent,
        radius=config.unbound_radius,
    )
    n_unbound_escapes = len(unbound_coords)
    # Frame-rate normalisation. The reweighted population is built from bin
    # COUNTS = t/dt (p*(r) = C t/dt, Eq. 3), so the two states must be expressed
    # at a common capture interval before they can be compared. The bound
    # interval is the reference; each unbound per-escape count (captured every
    # ``unbound_dt``) is rescaled to ``bound_dt`` by (unbound_dt / bound_dt) and
    # then raised to the UNBOUND exponent:
    #     frame_rate_factor = (unbound_dt / bound_dt) ** unbound_exponent.
    # NB: when the exponents differ (mixed-temperature two-state model) the ratio
    # retains a residual dependence on the reference interval ~ bound_dt **
    # (exponent - unbound_exponent); this is intrinsic (the Eq. 3 constant C no
    # longer cancels across states) and is pinned by keeping bound_frame_interval
    # at the paper's 0.01 ns.
    bound_dt = config.bound_frame_interval.value_in_unit(unit.picosecond)
    unbound_dt = config.unbound_frame_interval.value_in_unit(unit.picosecond)
    frame_rate_factor = (unbound_dt / bound_dt) ** unbound_exponent if bound_dt else 1.0
    unbound_population = unbound.total * frame_rate_factor
    unbound_ess = unbound.effective_sample_size

    if bound.total <= 0 or unbound_population <= 0:
        delta_g_comp = math.inf if bound.total <= 0 else -math.inf
    else:
        delta_g_comp = rt * math.log(unbound_population / bound.total)

    volume_correction = -rt * math.log(config.unbound_volume / config.standard_volume)
    delta_g = delta_g_comp + volume_correction

    # Sampled bound-well depth (diagnostic): reveals whether the bound trajectory
    # actually resolves a deep well or is capture-/sampling-limited.
    well = bound_well_diagnostics(
        bound_coords,
        bin_size=config.bin_size,
        boundary_radius=config.absorbing_boundary_radius,
        exponent=exponent,
        rt=rt,
    )

    return {
        "delta_g": delta_g,
        "delta_g_comp": delta_g_comp,
        "volume_correction": volume_correction,
        "bound_population": bound.total,
        "unbound_population": unbound_population,
        "n_bound_escapes": n_bound_escapes,
        "n_unbound_escapes": n_unbound_escapes,
        "bound_ess": bound.effective_sample_size,
        "bound_max_replica_fraction": bound.max_replica_fraction,
        "bound_population_min": (float(bound.per_replica.min()) if bound.per_replica.size else 0.0),
        "bound_population_max": (float(bound.per_replica.max()) if bound.per_replica.size else 0.0),
        "unbound_ess": unbound_ess,
        "c_min": well["c_min"],
        "c_boundary": well["c_boundary"],
        "c_min_radius": well["c_min_radius"],
        "delta_g_well": well["delta_g_well"],
    }


def compute_delta_g(
    bound_coords: list[np.ndarray],
    unbound_coords: list[np.ndarray],
    *,
    config: ModBindDGSettings,
    rt: float,
) -> dict:
    """Compute dG and per-replica independent estimates (Eq. 14)."""
    return _compute_delta_g_core(bound_coords, unbound_coords, config=config, rt=rt)


def bootstrap_delta_g(
    bound_coords: list[np.ndarray],
    unbound_coords: list[np.ndarray],
    *,
    config: ModBindDGSettings,
    rt: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap dG by resampling replicas; return (mean, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    n_bound = len(bound_coords)
    n_unbound = len(unbound_coords)
    if n_bound == 0:
        return math.nan, math.nan, math.nan

    samples: list[float] = []
    for _ in range(n_bootstrap):
        bound_sample = [bound_coords[i] for i in rng.integers(0, n_bound, n_bound)]
        if n_unbound > 0:
            unbound_sample = [unbound_coords[i] for i in rng.integers(0, n_unbound, n_unbound)]
        else:
            unbound_sample = unbound_coords
        result = compute_delta_g(bound_sample, unbound_sample, config=config, rt=rt)
        delta_g = result["delta_g"]
        if math.isfinite(delta_g):
            samples.append(delta_g)

    if not samples:
        return math.nan, math.nan, math.nan

    samples_arr = np.asarray(samples)
    return (
        float(samples_arr.mean()),
        float(np.percentile(samples_arr, 2.5)),
        float(np.percentile(samples_arr, 97.5)),
    )
