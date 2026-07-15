"""N-closest-waters MMGBSA on bound-state escape frames (COM < bound radius)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opensqm.md.mmgbsa import get_interaction_energy
from opensqm.modbind.states import LIGAND_RESNAME

if TYPE_CHECKING:
    from pathlib import Path

    from opensqm.modbind.config import ModBindDGSettings


def bound_dcd_frame_indices(displacements: np.ndarray, radius: float) -> np.ndarray:
    """Return DCD frame indices whose protein-referenced COM lies within ``radius``.

    DCD frame ``i`` corresponds to displacement row ``i + 1`` (row 0 is t=0).
    """
    com_dist = np.linalg.norm(displacements[1:], axis=1)
    return np.where(com_dist < radius)[0]


def compute_bound_mmgbsa(
    bound_trajectories: list[np.ndarray],
    *,
    trajectory_dir: Path,
    ligand_path: str,
    config: ModBindDGSettings,
    output_path: Path,
) -> dict[str, float | int]:
    """Run n-wat MMGBSA on all bound frames across escape replicas."""
    equil_pdb = trajectory_dir / "bound_equil.pdb"
    mmgbsa_dir = output_path / "mmgbsa"
    mmgbsa_dir.mkdir(parents=True, exist_ok=True)

    all_energies: list[float] = []
    n_bound_frames = 0

    for replica_i, displacements in enumerate(bound_trajectories):
        dcd_path = trajectory_dir / f"bound_{replica_i:04d}.dcd"
        frame_indices = bound_dcd_frame_indices(displacements, config.bound_state_radius)
        n_bound_frames += int(frame_indices.size)
        if frame_indices.size == 0:
            continue

        replica_dir = mmgbsa_dir / f"replica_{replica_i:04d}"
        replica_dir.mkdir(parents=True, exist_ok=True)
        close_traj = replica_dir / "bound.close.dcd"
        close_top = replica_dir / "bound.close.pdb"

        energies, _rmsd, _top, _traj = get_interaction_energy(
            pdb_path=str(equil_pdb),
            ligand_path=ligand_path,
            traj_path=str(dcd_path),
            close_traj_path=str(close_traj),
            close_top_path=str(close_top),
            n_closest_waters=config.n_closest_waters,
            ligand_resname=LIGAND_RESNAME,
            frame_indices=frame_indices,
        )
        all_energies.extend(float(e) for e in energies)
        # DCD frame ``j`` maps to displacement row ``j + 1`` (row 0 is t=0), so
        # each MMGBSA energy is paired with the COM coordinate that determines
        # its ModBinddG reweighting bin.

    arr = np.asarray(all_energies, dtype=np.float64)

    return {
        "mmgbsa_mean": float(arr.mean()),
        "mmgbsa_std": float(arr.std()) if arr.size else float("nan"),
        "mmgbsa_min": float(arr.min()) if arr.size else float("nan"),
        "mmgbsa_n_frames": n_bound_frames,
        "mmgbsa_n_decorrelated": int(arr.size),
    }
