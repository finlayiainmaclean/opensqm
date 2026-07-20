"""Collect ModBinddG escape trajectories with checkpointing and resume.

The unbound (ligand-in-solvent) trajectory is collected first as a single
``unbound.npz`` checkpoint, followed by ``n_replicas`` independent bound-state
escape trajectories saved as ``bound_NNNN.npy``. Existing checkpoints are
reused when ``resume`` is set so interrupted runs continue where they left off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from openmm import unit
from openmm.app import PDBFile
from tqdm import tqdm

from opensqm.modbind.escape import run_bound_escape, run_unbound_escape
from opensqm.modbind.reweight import compute_delta_g, rt_kcal

if TYPE_CHECKING:
    from opensqm.modbind.config import ModBindDGSettings
    from opensqm.modbind.states import PreparedState

UNBOUND_CHECKPOINT = "unbound.npz"


@dataclass
class ModBindDGData:
    """Raw escape sampling data passed to the analysis stage."""

    bound_trajectories: list[np.ndarray]
    bound_temperatures_k: list[float]
    temperature_k: float
    unbound_segments: list[np.ndarray] = field(default_factory=list)
    unbound_escape_times_steps: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    unbound_converged: bool = False


def _bound_checkpoint(checkpoint_dir: Path, index: int) -> Path:
    return checkpoint_dir / f"bound_{index:04d}.npy"


def _save_unbound_checkpoint(
    checkpoint_dir: Path,
    segments: list[np.ndarray],
    escape_times_steps: np.ndarray,
    converged: bool,
) -> None:
    payload: dict[str, np.ndarray] = {
        "escape_times_steps": np.asarray(escape_times_steps, dtype=np.int64),
        "converged": np.asarray(converged),
    }
    for i, segment in enumerate(segments):
        payload[f"escape_{i:04d}"] = segment
    np.savez(checkpoint_dir / UNBOUND_CHECKPOINT, **payload)


def _load_unbound_checkpoint(
    checkpoint_dir: Path,
) -> tuple[list[np.ndarray], np.ndarray, bool] | None:
    path = checkpoint_dir / UNBOUND_CHECKPOINT
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    segment_keys = sorted(
        k for k in data.keys() if k.startswith("escape_") and k[len("escape_") :].isdigit()
    )
    segments = [data[k] for k in segment_keys]
    escape_times_steps = data["escape_times_steps"]
    converged = bool(data["converged"])
    return segments, escape_times_steps, converged


def _bound_escape_time_ns(trajectory: np.ndarray, config: ModBindDGSettings, dt_ns: float) -> float:
    escaped = float(np.linalg.norm(trajectory[-1])) >= config.absorbing_boundary_radius
    sampled_ns = (len(trajectory) - 1) * dt_ns
    if escaped:
        return sampled_ns
    return config.max_escape_time.value_in_unit(unit.nanosecond)


def collect_trajectories(
    bound_state: PreparedState,
    unbound_state: PreparedState | None,
    config: ModBindDGSettings,
    *,
    checkpoint_dir: Path,
    trajectory_dir: Path,
    resume: bool = True,
) -> ModBindDGData:
    """Run (or resume) the unbound and bound escape simulations."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    unbound_segments: list[np.ndarray] = []
    unbound_escape_times_steps = np.empty(0, dtype=np.int64)
    unbound_converged = False

    cached = _load_unbound_checkpoint(checkpoint_dir) if resume else None
    if cached is not None:
        unbound_segments, unbound_escape_times_steps, unbound_converged = cached
        logger.info(f"Loaded {len(unbound_segments)} unbound escape segments from cache")
    else:
        if unbound_state is None:
            raise ValueError("A prepared unbound state is required")
        logger.info("Running unbound (ligand-in-solvent) escape simulation")
        unbound_segments, unbound_escape_times_steps, unbound_converged = run_unbound_escape(
            unbound_state, config
        )
        _save_unbound_checkpoint(
            checkpoint_dir,
            unbound_segments,
            unbound_escape_times_steps,
            unbound_converged,
        )

    trajectory_dir = Path(trajectory_dir)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    with (trajectory_dir / "bound_equil.pdb").open("w") as handle:
        PDBFile.writeFile(bound_state.topology, bound_state.positions, handle, keepIds=True)

    bound_trajectories: list[np.ndarray] = []
    dt_ns = config.bound_frame_interval.value_in_unit(unit.nanosecond)
    rt = rt_kcal(config.reference_temperature.value_in_unit(unit.kelvin))
    temperature_K = float(config.bound_temperature.value_in_unit(unit.kelvin))

    def _log_running_dg() -> None:
        """Log the pooled dG estimate after each bound replica finishes."""
        if not bound_trajectories:
            return
        try:
            partial = compute_delta_g(bound_trajectories, unbound_segments, config=config, rt=rt)
        except Exception as exc:
            logger.warning(f"Running dG estimate failed: {exc}")
            return
        dg = partial["delta_g"]
        tag = f"{len(bound_trajectories)}/{config.n_replicas} replicas"
        if math.isfinite(dg):
            logger.info(f"Running ΔG° estimate ({tag}): {dg:.2f} kcal/mol")
        else:
            logger.info(
                f"Running ΔG° estimate ({tag}): n/a (no bound-state population sampled yet)"
            )
        logger.info(
            f"  well: ΔG_well={partial['delta_g_well']:.2f} kcal/mol "
            f"(c_min={partial['c_min']} @ {partial['c_min_radius']:.1f} A, "
            f"c_boundary={partial['c_boundary']}); "
            f"P_bound={partial['bound_population']:.3g}, "
            f"P_unbound={partial['unbound_population']:.3g}"
        )

    with tqdm(total=config.n_replicas, desc="Bound escapes", unit="replica") as pbar:
        for index in range(config.n_replicas):
            checkpoint_path = _bound_checkpoint(checkpoint_dir, index)
            dcd_path = trajectory_dir / f"bound_{index:04d}.dcd"
            if resume and checkpoint_path.exists() and dcd_path.exists():
                trajectory = np.load(checkpoint_path)
                bound_trajectories.append(trajectory)
                escape_time_ns = _bound_escape_time_ns(trajectory, config, dt_ns)
                escaped = float(np.linalg.norm(trajectory[-1])) >= config.absorbing_boundary_radius
                verb = "escaped at" if escaped else "capped at"
                logger.info(
                    f"Bound replica {index} (cached): {verb} "
                    f"{escape_time_ns:.3f} ns at {temperature_K:.0f} K"
                )
                _log_running_dg()
                pbar.update(1)
                continue

            trajectory, diag = run_bound_escape(
                bound_state,
                config,
                seed=config.random_seed + index + 1,
                dcd_path=str(dcd_path),
                temperature=config.bound_temperature,
            )
            np.save(checkpoint_path, trajectory)
            bound_trajectories.append(trajectory)
            verb = "escaped at" if diag.escaped else "no escape, capped at"
            logger.info(
                f"Bound replica {index}: {verb} {diag.escape_time_ns:.3f} ns at "
                f"{temperature_K:.0f} K; "
                f"Calpha RMSD (last 10%) = {diag.calpha_rmsd_last10_a:.2f} A"
            )
            _log_running_dg()
            pbar.update(1)

    return ModBindDGData(
        bound_trajectories=bound_trajectories,
        bound_temperatures_k=[temperature_K] * len(bound_trajectories),
        temperature_k=temperature_K,
        unbound_segments=unbound_segments,
        unbound_escape_times_steps=unbound_escape_times_steps,
        unbound_converged=unbound_converged,
    )
