"""Collect ModBinddG escape trajectories with checkpointing and resume.

The unbound (ligand-in-solvent) trajectory is collected first as a single
``unbound.npz`` checkpoint, followed by ``n_replicas`` independent bound-state
escape trajectories saved as ``bound_NNNN.npy``. Existing checkpoints are
reused when ``resume`` is set so interrupted runs continue where they left off.
"""

from __future__ import annotations

import json
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
from opensqm.modbind.reweight import (
    compute_delta_g,
    predict_escape_temperature_calibrated,
    rt_kcal,
)

if TYPE_CHECKING:
    from opensqm.modbind.config import ModBindDGSettings
    from opensqm.modbind.states import PreparedState

UNBOUND_CHECKPOINT = "unbound.npz"
BOUND_TEMPERATURES_FILE = "bound_temperatures.json"


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
    unbound_mode: str = "explicit"


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


def _load_bound_temperatures(checkpoint_dir: Path) -> dict[int, float]:
    path = checkpoint_dir / BOUND_TEMPERATURES_FILE
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {int(k): float(v) for k, v in raw.items()}


def _save_bound_temperatures(checkpoint_dir: Path, temperatures: dict[int, float]) -> None:
    payload = {str(k): v for k, v in sorted(temperatures.items())}
    (checkpoint_dir / BOUND_TEMPERATURES_FILE).write_text(json.dumps(payload, indent=2))


def _analysis_config(
    config: ModBindDGSettings, bound_temperatures_k: list[float]
) -> ModBindDGSettings:
    if len(bound_temperatures_k) == config.n_replicas:
        return config.copy(
            update={
                "temperature": bound_temperatures_k[0] * unit.kelvin,
                "replica_temperatures": tuple(bound_temperatures_k),
            }
        )
    return config.copy(
        update={
            "n_replicas": len(bound_temperatures_k),
            "replica_temperatures": tuple(bound_temperatures_k),
        }
    )


def _bound_escape_time_ns(trajectory: np.ndarray, config: ModBindDGSettings, dt_ns: float) -> float:
    escaped = float(np.linalg.norm(trajectory[-1])) >= config.absorbing_boundary_radius
    sampled_ns = (len(trajectory) - 1) * dt_ns
    if escaped:
        return sampled_ns
    return config.max_escape_time.value_in_unit(unit.nanosecond)


def _mean_escape_temperature_for_target(
    *,
    bound_escape_times_ns: list[float],
    bound_temperatures_k: list[float],
    binding_dg_kcal: float,
    target_escape_time_ns: float | None,
    reference_temperature_k: float,
) -> float | None:
    """Mean calibrated T for ``target_escape_time_ns`` over finished replicas."""
    if target_escape_time_ns is None or target_escape_time_ns <= 0 or binding_dg_kcal >= 0:
        return None
    predictions: list[float] = []
    for tau_ns, temp_K in zip(bound_escape_times_ns, bound_temperatures_k, strict=False):
        if tau_ns <= 0 or temp_K <= 0:
            continue
        predicted = predict_escape_temperature_calibrated(
            temperature_k=temp_K,
            escape_time_ns=tau_ns,
            binding_dg_kcal=binding_dg_kcal,
            target_escape_time_ns=target_escape_time_ns,
            reference_temperature_k=reference_temperature_k,
        )
        if math.isfinite(predicted):
            predictions.append(predicted)
    if not predictions:
        return None
    return float(np.mean(predictions))


def _predict_next_replica_temperature(
    *,
    config: ModBindDGSettings,
    bound_trajectories: list[np.ndarray],
    bound_temperatures_k: list[float],
    bound_escape_times_ns: list[float],
    unbound_segments: list[np.ndarray],
    rt: float,
) -> float | None:
    if config.ideal_escape_time_ns is None or config.ideal_escape_time_ns <= 0:
        return None

    analysis_config = _analysis_config(config, bound_temperatures_k)
    result = compute_delta_g(bound_trajectories, unbound_segments, config=analysis_config, rt=rt)
    delta_g = float(result["delta_g"])

    if delta_g >= 0:
        delta_g = -3

    last_temperature_K = bound_temperatures_k[-1]
    t_room = config.reference_temperature.value_in_unit(unit.kelvin)
    target_tau_ns = config.ideal_escape_time_ns
    if not math.isfinite(delta_g) or delta_g >= 0:
        return last_temperature_K

    predicted = _mean_escape_temperature_for_target(
        bound_escape_times_ns=bound_escape_times_ns,
        bound_temperatures_k=bound_temperatures_k,
        binding_dg_kcal=delta_g,
        target_escape_time_ns=target_tau_ns,
        reference_temperature_k=t_room,
    )
    if predicted is None or not math.isfinite(predicted):
        return last_temperature_K

    return np.clip(predicted, 650.0, 1200.0)


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

    if config.unbound_mode == "explicit":
        cached = _load_unbound_checkpoint(checkpoint_dir) if resume else None
        if cached is not None:
            unbound_segments, unbound_escape_times_steps, unbound_converged = cached
            logger.info(f"Loaded {len(unbound_segments)} unbound escape segments from cache")
        else:
            if unbound_state is None:
                raise ValueError("Explicit unbound mode requires a prepared unbound state")
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
    bound_temperatures_k: list[float] = []
    bound_escape_times_ns: list[float] = []
    stored_temperatures = _load_bound_temperatures(checkpoint_dir) if resume else {}
    dt_ns = config.bound_frame_interval.value_in_unit(unit.nanosecond)
    rt = rt_kcal(config.reference_temperature.value_in_unit(unit.kelvin))
    adaptive = config.ideal_escape_time_ns is not None and config.ideal_escape_time_ns > 0
    fixed_temperatures = None if adaptive else list(config.bound_temperatures_k())
    seed_temperature_K = float(config.temperature.value_in_unit(unit.kelvin))

    def _log_running_dg(replica_index: int) -> None:
        """Log pooled and single-replica dG estimates after one replica finishes."""
        if not bound_trajectories:
            return
        try:
            analysis_config = _analysis_config(config, bound_temperatures_k)
            partial = compute_delta_g(
                bound_trajectories, unbound_segments, config=analysis_config, rt=rt
            )
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
        if (
            math.isfinite(dg)
            and dg < 0
            and config.ideal_escape_time_ns is not None
            and config.ideal_escape_time_ns > 0
        ):
            t_room = config.reference_temperature.value_in_unit(unit.kelvin)
            t_for_1ns = _mean_escape_temperature_for_target(
                bound_escape_times_ns=bound_escape_times_ns,
                bound_temperatures_k=bound_temperatures_k,
                binding_dg_kcal=dg,
                target_escape_time_ns=config.ideal_escape_time_ns,
                reference_temperature_k=t_room,
            )
            if t_for_1ns is not None:
                logger.info(
                    f"Estimated T for {config.ideal_escape_time_ns} ns escape: {t_for_1ns:.0f} K "
                    f"(mean over {len(bound_escape_times_ns)} replica(s), "
                )

    with tqdm(total=config.n_replicas, desc="Bound escapes", unit="replica") as pbar:
        for index in range(config.n_replicas):
            if fixed_temperatures is not None:
                replica_temperature_K = fixed_temperatures[index]
            elif index == 0:
                replica_temperature_K = seed_temperature_K
            else:
                predicted = _predict_next_replica_temperature(
                    config=config,
                    bound_trajectories=bound_trajectories,
                    bound_temperatures_k=bound_temperatures_k,
                    bound_escape_times_ns=bound_escape_times_ns,
                    unbound_segments=unbound_segments,
                    rt=rt,
                )
                replica_temperature_K = predicted or bound_temperatures_k[-1]

            replica_temperature = replica_temperature_K * unit.kelvin
            checkpoint_path = _bound_checkpoint(checkpoint_dir, index)
            dcd_path = trajectory_dir / f"bound_{index:04d}.dcd"
            if resume and checkpoint_path.exists() and dcd_path.exists():
                trajectory = np.load(checkpoint_path)
                bound_trajectories.append(trajectory)
                cached_temperature_K = stored_temperatures.get(index)
                if cached_temperature_K is None:
                    cached_temperature_K = (
                        fixed_temperatures[index]
                        if fixed_temperatures is not None
                        else seed_temperature_K
                    )
                bound_temperatures_k.append(cached_temperature_K)
                escape_time_ns = _bound_escape_time_ns(trajectory, config, dt_ns)
                bound_escape_times_ns.append(escape_time_ns)
                escaped = float(np.linalg.norm(trajectory[-1])) >= config.absorbing_boundary_radius
                verb = "escaped at" if escaped else "capped at"
                logger.info(
                    f"Bound replica {index} (cached): {verb} "
                    f"{escape_time_ns:.3f} ns at {cached_temperature_K:.0f} K"
                )
                _log_running_dg(index)
                pbar.update(1)
                continue

            dcd_path = str(dcd_path)
            trajectory, diag = run_bound_escape(
                bound_state,
                config,
                seed=config.random_seed + index + 1,
                dcd_path=dcd_path,
                temperature=replica_temperature,
            )
            np.save(checkpoint_path, trajectory)
            bound_trajectories.append(trajectory)
            bound_temperatures_k.append(replica_temperature_K)
            bound_escape_times_ns.append(diag.escape_time_ns)
            stored_temperatures[index] = replica_temperature_K
            _save_bound_temperatures(checkpoint_dir, stored_temperatures)
            verb = "escaped at" if diag.escaped else "no escape, capped at"
            logger.info(
                f"Bound replica {index}: {verb} {diag.escape_time_ns:.3f} ns at "
                f"{replica_temperature_K:.0f} K; "
                f"Calpha RMSD (last 10%) = {diag.calpha_rmsd_last10_a:.2f} A"
            )
            _log_running_dg(index)
            pbar.update(1)

    return ModBindDGData(
        bound_trajectories=bound_trajectories,
        bound_temperatures_k=bound_temperatures_k,
        temperature_k=seed_temperature_K,
        unbound_segments=unbound_segments,
        unbound_escape_times_steps=unbound_escape_times_steps,
        unbound_converged=unbound_converged,
        unbound_mode=config.unbound_mode,
    )
