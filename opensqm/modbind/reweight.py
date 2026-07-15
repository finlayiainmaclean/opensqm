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


def predict_escape_temperature_calibrated(
    *,
    temperature_k: float,
    escape_time_ns: float,
    binding_dg_kcal: float,
    target_escape_time_ns: float | None,
    reference_temperature_k: float = 300.0,
    min_temperature_k: float = 300.0,
    max_temperature_k: float = 2000.0,
) -> float:
    """Predict the simulation temperature for a target escape time.

    Inverts ``ln(τ₂/τ₁) = (|ΔG°|/RT_room) (T₁/T₂ - 1)``, which matches the
    empirical observation that ligands with |ΔG°|≈5.5 kcal/mol escape in
    roughly 1-4 ns at 650 K. ``binding_dg_kcal`` must be negative (favourable
    binding); weaker (less negative) estimates give lower temperatures for the
    same target escape time.
    """
    if (
        not math.isfinite(binding_dg_kcal)
        or escape_time_ns <= 0
        or target_escape_time_ns is None
        or target_escape_time_ns <= 0
        or temperature_k <= 0
        or binding_dg_kcal >= 0
    ):
        return math.nan

    rt_room = rt_kcal(reference_temperature_k)
    if rt_room <= 0:
        return math.nan

    well_depth = -binding_dg_kcal
    scale = well_depth / rt_room
    log_ratio = math.log(target_escape_time_ns / escape_time_ns)
    denom = 1.0 + log_ratio / scale
    if denom <= 0:
        return math.nan

    predicted = temperature_k / denom
    return float(np.clip(predicted, min_temperature_k, max_temperature_k))


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
    (Angstrom). All frames are pooled into a single cubic histogram; bin counts
    are raised to a per-replica ``exponent`` and shared equally among the frames
    in each bin.
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

    per_replica = np.zeros(len(coords_list), dtype=np.float64)
    for replica_i, (coords, keys, exp) in enumerate(
        zip(coords_list, keys_per_replica, exponents, strict=False)
    ):
        radii = np.linalg.norm(np.asarray(coords, dtype=np.float64), axis=1)
        population = 0.0
        for r, key in zip(radii, keys, strict=False):
            if r <= radius:
                count = counts[key]
                population += float(count) ** exp / count
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

    n_shells = max(1, math.ceil(max_radius / bin_size))
    shell_pop = np.zeros(n_shells, dtype=np.float64)
    for keys, radii, exp in zip(keys_per_replica, radii_per_replica, exponents, strict=False):
        for r, key in zip(radii, keys, strict=False):
            shell = int(r // bin_size)
            if 0 <= shell < n_shells:
                count = counts[key]
                shell_pop[shell] += float(count) ** exp / count

    centers = (np.arange(n_shells) + 0.5) * bin_size
    with np.errstate(divide="ignore"):
        pmf = -rt * np.log(shell_pop)
    finite = np.isfinite(pmf)
    if finite.any():
        pmf[finite] -= pmf[finite].min()
    return centers, pmf


def einstein_smoluchowski_unbound(
    config: ModBindDGSettings,
    *,
    rt: float,
    n_bound_escapes: int | None = None,
) -> tuple[float, float]:
    """Estimate the unbound population and free energy (SI Eq. SI-1..SI-4).

    Treats the ligand as a hard sphere diffusing with coefficient ``D``; the
    mean time to diffuse ``r`` is ``t = r^2 / (6 D)``. The unbound "population"
    matches the bound-state simulation parameters: ``N_escapes * t /
    frame_interval``, where ``N_escapes`` is the number of bound replicas
    included in the current population ratio (defaults to ``config.n_replicas``).
    """
    from openmm import unit

    radius_m = config.einstein_radius * 1e-10
    t_seconds = radius_m**2 / (6.0 * config.diffusion_coefficient_m2_s)
    t_ns = t_seconds * 1e9
    # SI Eq. SI-4 applies the bound simulation's frame-capture interval.
    frame_interval_ns = config.bound_frame_interval.value_in_unit(unit.nanosecond)

    n_escapes = config.n_replicas if n_bound_escapes is None else n_bound_escapes
    population = n_escapes * t_ns / frame_interval_ns
    g = -rt * math.log(population) if population > 0 else math.inf
    return population, g


def _compute_delta_g_core(
    bound_coords: list[np.ndarray],
    unbound_coords: list[np.ndarray],
    *,
    config: ModBindDGSettings,
    rt: float,
) -> dict:
    """Compute pooled dG and intermediate terms from reweighted populations."""
    bound = reweight_state(
        bound_coords,
        bin_size=config.bin_size,
        exponent=config.bound_reweight_exponents(),
        radius=config.bound_state_radius,
    )

    n_bound_escapes = len(bound_coords)
    unbound_g = None
    if config.unbound_mode == "einstein":
        unbound_population, unbound_g = einstein_smoluchowski_unbound(
            config, rt=rt, n_bound_escapes=n_bound_escapes
        )
        n_unbound_escapes = 0
        unbound_ess = 0.0
    else:
        unbound_exponent = config.unbound_temperature.value_in_unit(
            unit.kelvin
        ) / config.reference_temperature.value_in_unit(unit.kelvin)
        unbound = reweight_state(
            unbound_coords,
            bin_size=config.bin_size,
            exponent=unbound_exponent,
            radius=config.unbound_radius,
        )
        n_unbound_escapes = len(unbound_coords)
        # Flux balance: equalise the number of escape events between states.
        flux_factor = n_bound_escapes / n_unbound_escapes if n_unbound_escapes else 1.0
        # Frame-rate normalisation: populations are time/dt (Eq. 3), so rescale
        # the unbound counts (captured at a different interval) onto the bound
        # state's frame-capture interval before forming the ratio.
        bound_dt = config.bound_frame_interval.value_in_unit(unit.picosecond)
        unbound_dt = config.unbound_frame_interval.value_in_unit(unit.picosecond)
        frame_rate_factor = unbound_dt / bound_dt if bound_dt else 1.0
        unbound_population = unbound.total * flux_factor * frame_rate_factor
        unbound_ess = unbound.effective_sample_size

    if bound.total <= 0 or unbound_population <= 0:
        delta_g_comp = math.inf if bound.total <= 0 else -math.inf
    else:
        delta_g_comp = rt * math.log(unbound_population / bound.total)

    volume_correction = -rt * math.log(config.unbound_volume / config.standard_volume)
    delta_g = delta_g_comp + volume_correction

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
        "unbound_g": unbound_g,
        "unbound_ess": unbound_ess,
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
        if config.unbound_mode == "explicit" and n_unbound > 0:
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
