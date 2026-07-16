"""pH-stratified and overall MMGBSA on constant-pH trajectories."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from openmm.app import PDBFile

from opensqm.cph.trajectory import (
    iter_replica_state_trajectories,
    replica_trajectory_index,
)
from opensqm.md.charge_cache import ligand_variant_path, load_ligand_variant
from opensqm.md.mmgbsa import get_interaction_energy


def _ligand_resname_from_pdb(pdb_path: Path) -> str:
    for res in PDBFile(str(pdb_path)).topology.residues():
        if res.name == "LIG" or res.name.startswith("LIG"):
            return res.name
    raise ValueError(f"No ligand residue found in {pdb_path}")


def _resolve_ligand_sdf(
    output_path: Path,
    resname: str,
    ligand_path: str,
) -> Path:
    cached = ligand_variant_path(output_path, resname)
    if cached.exists():
        return cached
    return Path(ligand_path).expanduser().resolve()


def _snap_ph(ph: float, ph_ladder: list[float] | None) -> float:
    if ph_ladder:
        return min(ph_ladder, key=lambda p: abs(p - ph))
    return round(ph, 4)


def _batch_index_for_step(step: int, protonation_swap_steps: int, n_batches: int) -> int:
    if n_batches <= 0:
        return 0
    return min(max(0, (step - 1) // protonation_swap_steps), n_batches - 1)


def _frame_indices_by_ph(
    traj_csv: pd.DataFrame,
    state_label: str,
    batch_ph: pd.Series,
    protonation_swap_steps: int,
    ph_ladder: list[float] | None,
) -> dict[float, np.ndarray]:
    """Group DCD frame indices by the replica pH active when each frame was written."""
    state_rows = traj_csv.loc[traj_csv["system_state"] == state_label]
    grouped: dict[float, list[int]] = defaultdict(list)
    n_batches = len(batch_ph)

    for _, row in state_rows.iterrows():
        step = int(row["step"])
        frame_ix = int(row["frame_ix"])
        batch_idx = _batch_index_for_step(step, protonation_swap_steps, n_batches)
        ph = _snap_ph(float(batch_ph.iloc[batch_idx]), ph_ladder)
        grouped[ph].append(frame_ix)

    return {
        ph: np.asarray(sorted(set(indices)), dtype=np.int64)
        for ph, indices in grouped.items()
        if indices
    }


def _summarize_energies(energies: list[float]) -> dict[str, float | int]:
    arr = np.asarray(energies, dtype=np.float64)
    if arr.size == 0:
        return {
            "mmgbsa_mean": float("nan"),
            "mmgbsa_std": float("nan"),
            "mmgbsa_min": float("nan"),
            "mmgbsa_n_frames": 0,
        }
    return {
        "mmgbsa_mean": float(arr.mean()),
        "mmgbsa_std": float(arr.std()) if arr.size > 1 else 0.0,
        "mmgbsa_min": float(arr.min()),
        "mmgbsa_n_frames": int(arr.size),
    }


def compute_replica_mmgbsa(
    output_path: Path,
    ligand_path: str,
    replica_dfs: list[pd.DataFrame],
    *,
    protonation_swap_steps: int,
    ph_ladder: list[float] | None = None,
    n_closest_waters: int = 5,
    score_phs: list[float] | None = None,
) -> dict[str, Any]:
    """Run pH-stratified and overall n-closest-waters MMGBSA across all replicas.

    When ``score_phs`` is given, only frames whose active pH (snapped to
    ``ph_ladder``) falls in that set are scored. MMGBSA is pH-invariant for a
    fixed protonation microstate, so scoring a single target pH still yields the
    correct per-microstate energies while skipping the (expensive)
    interaction-energy evaluation on the other ladder rungs. ``None`` scores
    every pH sampled.
    """
    mmgbsa_dir = output_path / "mmgbsa"
    mmgbsa_dir.mkdir(parents=True, exist_ok=True)
    close_traj = mmgbsa_dir / "_close.dcd"
    close_top = mmgbsa_dir / "_close.pdb"

    score_ph_set = {_snap_ph(p, ph_ladder) for p in score_phs} if score_phs is not None else None

    ph_summaries: dict[float, list[float]] = defaultdict(list)
    all_energies: list[float] = []
    frame_records: list[dict] = []

    # Pre-load charged molecules once per unique resname to avoid re-running AM1BCC.
    charged_offmols: dict[str, object] = {}

    for replica_i, replica_df in enumerate(replica_dfs):
        # Force ``system_state`` to string: with a single titratable residue the
        # column holds only single integers ("0"/"1"), which pandas would infer
        # as int64, so the ``== state_label`` (a string from the DCD filename)
        # comparison below would match nothing and silently yield zero frames.
        traj_csv = pd.read_csv(
            replica_trajectory_index(output_path, replica_i),
            dtype={"system_state": str},
        )
        batch_ph = replica_df["ph"]

        for dcd_path, pdb_path in iter_replica_state_trajectories(output_path, replica_i):
            resname = _ligand_resname_from_pdb(pdb_path)
            ligand_sdf = _resolve_ligand_sdf(output_path, resname, ligand_path)
            if resname not in charged_offmols:
                charged_offmols[resname] = load_ligand_variant(output_path, resname)
            offmol = charged_offmols[resname]
            state_label = dcd_path.name.split(".", 1)[-1].removesuffix(".dcd")
            ph_groups = _frame_indices_by_ph(
                traj_csv,
                state_label,
                batch_ph,
                protonation_swap_steps,
                ph_ladder,
            )
            if score_ph_set is not None:
                ph_groups = {ph: idx for ph, idx in ph_groups.items() if ph in score_ph_set}
            if not ph_groups:
                continue
            state_rows = traj_csv[traj_csv["system_state"] == state_label]
            frame_to_time = dict(
                zip(state_rows["frame_ix"].astype(int), state_rows["time_ns"], strict=False)
            )

            for ph, frame_indices in sorted(ph_groups.items()):
                logger.info(
                    f"MMGBSA replica {replica_i} state {state_label} pH {ph:.3f} "
                    f"({len(frame_indices)} raw frames, {resname})"
                )
                energies, _rmsd, _top, _traj = get_interaction_energy(
                    pdb_path=str(pdb_path),
                    ligand_path=str(ligand_sdf),
                    traj_path=str(dcd_path),
                    close_traj_path=str(close_traj),
                    close_top_path=str(close_top),
                    n_closest_waters=n_closest_waters,
                    ligand_resname=resname,
                    frame_indices=frame_indices,
                    offmol=offmol,
                )
                ph_summaries[ph].extend(float(e) for e in energies)
                all_energies.extend(float(e) for e in energies)
                for j, frame_ix in enumerate(frame_indices):
                    frame_records.append(
                        {
                            "ph": ph,
                            "state_label": state_label,
                            "time_ns": frame_to_time.get(int(frame_ix), float("nan")),
                            "replica_i": replica_i,
                            "energy": float(energies[j]),
                            "energy_type": "mmgbsa",
                            "dcd_path": dcd_path,
                            "pdb_path": pdb_path,
                            "frame_ix": int(frame_ix),
                        }
                    )

    by_ph_rows = [
        {"ph": ph, **_summarize_energies(ph_summaries[ph])} for ph in sorted(ph_summaries)
    ]
    pd.DataFrame(by_ph_rows).to_csv(mmgbsa_dir / "by_ph.csv", index=False)
    for row in by_ph_rows:
        logger.info(
            f"pH {row['ph']:.3f} MMGBSA = {row['mmgbsa_mean']:.2f} "
            f"+/- {row['mmgbsa_std']:.2f} kcal/mol "
            f"({row['mmgbsa_n_frames']} decorrelated frames)"
        )

    overall = _summarize_energies(all_energies)
    pd.DataFrame([overall]).to_csv(mmgbsa_dir / "overall.csv", index=False)
    logger.info(
        f"Overall MMGBSA = {overall['mmgbsa_mean']:.2f} "
        f"+/- {overall['mmgbsa_std']:.2f} kcal/mol "
        f"({overall['mmgbsa_n_frames']} decorrelated frames)"
    )

    # The n-closest-waters complex is scratch that get_interaction_energy rewrites
    # for every (replica, state, pH) group; only the energies are kept.
    close_traj.unlink(missing_ok=True)
    close_top.unlink(missing_ok=True)

    return {
        "mmgbsa_by_ph": by_ph_rows,
        "mmgbsa_frames": frame_records,
        **{f"mmgbsa_{k}": v for k, v in overall.items()},
    }
