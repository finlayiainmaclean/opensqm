"""Population-based reweighting and free-energy estimation for ModBinddG.

Implements the discrete form of Eq. 14 of Sinko et al. (*PNAS* 2026):

    dG = RT ln( sum_unbound (p*)^(1/lambda) / sum_bound (p*)^(1/lambda) )
         - RT ln(Vu / V0)

Configurations are binned into equal-volume Cartesian cubes and bin counts are
raised to ``1/lambda`` (``lambda = T_ref / T_sim``) to reweight the
high-temperature populations back to the reference temperature.

Each escape trajectory is binned and reweighted INDEPENDENTLY, and the
per-trajectory reweighted populations are then averaged over trajectories -- the
per-trajectory-mean form the paper derives via the law of large numbers (Sinko
et al. SI Eqs. SI-11/SI-12, Fig. S13):
``P_state = (1/N) * sum_k sum_bins count[bin, k] ** (1/lambda)``. Averaging over
``N`` -- rather than pooling all trajectories into one histogram and dividing the
POOLED count by ``N`` before the exponent -- is what makes each state population
intensive (invariant to the number of escapes for ANY exponent) and matches the
paper's normalisation exactly. The two forms agree only when every trajectory
occupies the same bins; when trajectories spread across bins, pooling-then-
dividing undercounts the ``(.)^(1/lambda)`` sum by up to ``RT ln N``.

The bound and unbound states are reweighted with their OWN exponents
(``T_state / T_ref``), because the paper runs them at different temperatures: a
hot bound (unbinding) simulation and a 300 K unbound (ligand-in-solvent)
simulation whose exponent is therefore 1 (no reweighting). Because each state is
its own per-trajectory mean, the Eq. 14 ratio is unaffected when the two states
are run for a *different* number of escapes (e.g. 8 bound vs 32 unbound), which
the paper permits: "it is also acceptable to normalize the populations of each
state if different numbers of escape simulations are performed for either state."
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
    """Reweight each escape trajectory independently and average over trajectories.

    ``coords_list`` is a list of ``(n_frames, 3)`` COM-displacement arrays
    (Angstrom). Each trajectory is binned on its own into cubic bins; its in-state
    (radius <= ``radius``) per-bin counts are raised to that replica's
    ``exponent``, summed over bins, and divided by the number of trajectories.
    The returned ``total`` is the sum of these per-trajectory contributions --
    i.e. the MEAN per-trajectory reweighted population (Sinko et al. SI Eqs.
    SI-11/SI-12). This, rather than pooling all trajectories then dividing the
    pooled count by ``N`` before the exponent, is the paper's normalisation and
    is what keeps the population intensive (escape-count invariant) for ANY
    exponent.
    """
    if not coords_list:
        return StatePopulation(total=0.0, per_replica=np.empty(0))

    exponents = _replica_exponents(exponent, len(coords_list))
    n_replicas = len(coords_list)
    per_replica = np.zeros(n_replicas, dtype=np.float64)
    for replica_i, (coords, exp) in enumerate(zip(coords_list, exponents, strict=False)):
        arr = np.asarray(coords, dtype=np.float64)
        radii = np.linalg.norm(arr, axis=1)
        mask = radii <= radius
        if not mask.any():
            continue
        bin_idx = np.floor(arr[mask] / bin_size).astype(int)
        counts: dict[tuple[int, int, int], int] = {}
        for row in bin_idx:
            key = (int(row[0]), int(row[1]), int(row[2]))
            counts[key] = counts.get(key, 0) + 1
        # Per-trajectory reweighted population: sum of count**exponent over this
        # trajectory's own in-state bins, divided by the number of trajectories so
        # the pooled total is the mean per-trajectory population (Eq. SI-12).
        per_replica[replica_i] = sum(c**exp for c in counts.values()) / n_replicas

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

    Uses the same per-trajectory-mean reweighting as :func:`reweight_state` (each
    trajectory binned independently, per-bin counts raised to the replica
    exponent, averaged over trajectories), then aggregates the reweighted bin
    populations into radial shells of width ``bin_size``.
    """
    if not coords_list:
        return np.empty(0), np.empty(0)

    exponents = _replica_exponents(exponent, len(coords_list))
    n_replicas = len(coords_list)
    n_shells = max(1, math.ceil(max_radius / bin_size))
    shell_pop = np.zeros(n_shells, dtype=np.float64)
    for coords, exp in zip(coords_list, exponents, strict=False):
        arr = np.asarray(coords, dtype=np.float64)
        radii = np.linalg.norm(arr, axis=1)
        bin_idx = np.floor(arr / bin_size).astype(int)
        keys = [(int(row[0]), int(row[1]), int(row[2])) for row in bin_idx]
        counts: dict[tuple[int, int, int], int] = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        # Each bin contributes count**exponent to its shell, shared equally over
        # the bin's frames (count**(exponent-1) per frame) and averaged over
        # trajectories (/ n_replicas), matching :func:`reweight_state`.
        for r, key in zip(radii, keys, strict=False):
            shell = int(r // bin_size)
            if 0 <= shell < n_shells:
                shell_pop[shell] += counts[key] ** (exp - 1.0) / n_replicas

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
    well_radius: float,
) -> dict:
    """Diagnose the sampled bound-well depth from pooled bound trajectories.

    Bins all bound frames into the same cubic histogram used by
    :func:`reweight_state` and reports, purely as diagnostics:

    * ``c_min`` -- occupancy of the most-populated bin inside the bound well
      (mean COM radius <= ``well_radius``). 0 if the ligand never dwells in the
      well (e.g. ballistic escape at too-high temperature).
    * ``c_boundary`` -- occupancy of the most-populated bin in a tight shell
      (+/- half a bin) around the absorbing boundary -- the free-ligand reference.
    * ``c_min_radius`` -- mean COM radius (Angstrom) of the well-bottom bin.
    * ``delta_g_well`` -- the well depth this sampling can support,
      ``-exponent * RT * ln(c_min / c_boundary)``. Boundary-anchored, so it is
      independent of replica count and frame rate, but bounded in magnitude by
      ``exponent * RT * ln(frames_per_escape)``. If it is far shallower than the
      expected affinity, the bound well is under-sampled -- the ligand escaped
      before dwelling (temperature too high) or the boundary transit is faster
      than one ``bound_frame_interval`` (capture too coarse).
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

    nan_result = {"c_min": 0, "c_boundary": 0, "c_min_radius": math.nan, "delta_g_well": math.nan}
    if not counts:
        return nan_result

    mean_radius = {k: radius_sum[k] / counts[k] for k in counts}

    # Well bottom: most-occupied bin whose frames sit inside the bound well.
    well_bins = {k: c for k, c in counts.items() if mean_radius[k] <= well_radius}
    # Boundary reference: most-occupied bin in a tight shell around the boundary.
    half = bin_size / 2.0
    boundary_bins = {
        k: c for k, c in counts.items() if abs(mean_radius[k] - boundary_radius) <= half
    }
    if not well_bins or not boundary_bins:
        # Well or boundary shell unsampled -> depth is undefined (report counts).
        c_min = max(well_bins.values()) if well_bins else 0
        c_boundary = max(boundary_bins.values()) if boundary_bins else 0
        c_min_key = max(well_bins, key=lambda k: well_bins[k]) if well_bins else None
        return {
            "c_min": int(c_min),
            "c_boundary": int(c_boundary),
            "c_min_radius": float(mean_radius[c_min_key]) if c_min_key else math.nan,
            "delta_g_well": math.nan,
        }

    c_min_key = max(well_bins, key=lambda k: well_bins[k])
    c_min = well_bins[c_min_key]
    c_boundary = max(boundary_bins.values())

    delta_g_well = -exponent * rt * math.log(c_min / c_boundary)

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
        well_radius=config.bound_state_radius,
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
