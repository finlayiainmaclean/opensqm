"""Analysis stage for ModBinddG: reweighting, dG, PMF, and output files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from openmm import unit

from opensqm.modbinddg.mmgbsa import compute_bound_mmgbsa
from opensqm.modbinddg.reweight import (
    bootstrap_delta_g,
    compute_delta_g,
    radial_pmf,
    rt_kcal,
)

if TYPE_CHECKING:
    from opensqm.modbinddg.config import ModBindDGSettings
    from opensqm.modbinddg.simulate import ModBindDGData

RESULTS_COLUMNS = [
    "delta_g",
    "delta_g_bootstrap_mean",
    "ci_low",
    "ci_high",
    "delta_g_comp",
    "volume_correction",
    "bound_population",
    "unbound_population",
    "n_bound_trajectories",
    "n_unbound_trajectories",
    "n_bound_escapes",
    "n_unbound_escapes",
    "unbound_converged",
    "unbound_mean_escape_time_ns",
    "bound_ess",
    "bound_max_replica_fraction",
    "bound_population_min",
    "bound_population_max",
    "unbound_mode",
    "unbound_g",
    "temperature_K",
    "mmgbsa_mean",
    "mmgbsa_std",
    "mmgbsa_min",
    "mmgbsa_n_frames",
    "mmgbsa_n_decorrelated",
    "n_closest_waters",
    "bound_sim_time_ns",
    "unbound_sim_time_ns",
    "total_sim_time_ns",
]


def _write_pmf(
    bound_coords: list[np.ndarray],
    config: ModBindDGSettings,
    rt: float,
    output_path: Path,
) -> None:
    centers, pmf = radial_pmf(
        bound_coords,
        bin_size=config.bin_size,
        exponent=config.bound_reweight_exponents(),
        max_radius=config.pmf_max_radius,
        rt=rt,
    )
    if centers.size:
        finite = np.isfinite(pmf)
        last_finite = int(np.max(np.where(finite))) if finite.any() else -1
        keep = min(centers.size, last_finite + 2)
        centers, pmf = centers[:keep], pmf[:keep]
    pd.DataFrame({"radius": centers, "pmf": pmf}).to_csv(
        output_path / "pmf.csv", index=False
    )


def analyze_modbinddg(
    data: ModBindDGData,
    config: ModBindDGSettings,
    output_path: Path,
    *,
    ligand_path: str,
    trajectory_dir: Path,
) -> dict:
    """Compute dG (Eq. 14), bootstrap CI, PMF, and write results.csv/pmf.csv."""
    output_path = Path(output_path)
    rt = rt_kcal(config.reference_temperature.value_in_unit(unit.kelvin))

    bound_coords = data.bound_trajectories
    unbound_coords = data.unbound_segments

    point = compute_delta_g(bound_coords, unbound_coords, config=config, rt=rt)
    bootstrap_mean, ci_low, ci_high = bootstrap_delta_g(
        bound_coords,
        unbound_coords,
        config=config,
        rt=rt,
        n_bootstrap=config.n_bootstrap,
        seed=config.random_seed,
    )

    step_size_ps = config.integrator_step_size.value_in_unit(unit.picosecond)
    escape_times = np.asarray(data.unbound_escape_times_steps, dtype=np.float64)
    unbound_mean_escape_time_ns = (
        float(escape_times.mean()) * step_size_ps / 1000.0 if escape_times.size else None
    )

    bound_dt_ns = config.bound_frame_interval.value_in_unit(unit.nanosecond)
    bound_sim_time_ns = float(
        sum(max(len(traj) - 1, 0) for traj in bound_coords) * bound_dt_ns
    )
    unbound_sim_time_ns = float(escape_times.sum() * step_size_ps / 1000.0)
    total_sim_time_ns = bound_sim_time_ns + unbound_sim_time_ns

    _write_pmf(bound_coords, config, rt, output_path)

    logger.info("Computing bound-state MMGBSA (COM < bound radius)")
    mmgbsa = compute_bound_mmgbsa(
        bound_coords,
        trajectory_dir=trajectory_dir,
        ligand_path=ligand_path,
        config=config,
        output_path=output_path,
    )

    results = {
        "delta_g": point["delta_g"],
        "delta_g_bootstrap_mean": bootstrap_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "delta_g_comp": point["delta_g_comp"],
        "volume_correction": point["volume_correction"],
        "bound_population": point["bound_population"],
        "unbound_population": point["unbound_population"],
        "n_bound_trajectories": len(bound_coords),
        "n_unbound_trajectories": len(unbound_coords),
        "n_bound_escapes": point["n_bound_escapes"],
        "n_unbound_escapes": point["n_unbound_escapes"],
        "unbound_converged": bool(data.unbound_converged),
        "unbound_mean_escape_time_ns": unbound_mean_escape_time_ns,
        "bound_ess": point["bound_ess"],
        "bound_max_replica_fraction": point["bound_max_replica_fraction"],
        "bound_population_min": point["bound_population_min"],
        "bound_population_max": point["bound_population_max"],
        "unbound_mode": config.unbound_mode,
        "unbound_g": point["unbound_g"],
        "temperature_K": data.temperature_K,
        "bound_sim_time_ns": bound_sim_time_ns,
        "unbound_sim_time_ns": unbound_sim_time_ns,
        "total_sim_time_ns": total_sim_time_ns,
        **mmgbsa,
    }

    pd.DataFrame([results], columns=RESULTS_COLUMNS).to_csv(
        output_path / "results.csv", index=False
    )
    logger.info(
        f"dG = {results['delta_g']:.2f} "
        f"[{results['ci_low']:.2f}, {results['ci_high']:.2f}] kcal/mol"
    )

    logger.info(
        f"Bound MMGBSA = {results['mmgbsa_mean']:.2f} "
        f"+/- {results['mmgbsa_std']:.2f} kcal/mol "
        f"{results['mmgbsa_n_frames']} bound frames)"
    )
    return results
