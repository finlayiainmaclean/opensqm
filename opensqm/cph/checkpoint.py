"""Checkpoint and manifest I/O for constant-pH REMD runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from opensqm.cph.simulation_config import ConstantpHSettings

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "run_manifest.json"
EQUILIBRATED_PDB = "equilibrated.pdb"
CHECKPOINT_DIRNAME = "checkpoints"
REMD_STATE_FILENAME = "remd_state.json"
PRODUCTION_STATE_FILENAME = "production_state.json"


def manifest_path(output_path: Path) -> Path:
    """Return the run-manifest JSON path under ``output_path``."""
    return output_path / MANIFEST_FILENAME


def checkpoint_dir(output_path: Path) -> Path:
    """Return the checkpoints directory under ``output_path``."""
    return output_path / CHECKPOINT_DIRNAME


def equilibrated_pdb_path(output_path: Path) -> Path:
    """Return the equilibrated-system PDB path under ``output_path``."""
    return output_path / EQUILIBRATED_PDB


def ph_ladder(min_ph: float, max_ph: float, ph_spacing: float) -> list[float]:
    """Return an inclusive pH ladder from ``min_ph`` to ``max_ph`` at ``ph_spacing``."""
    import numpy as np

    return [float(ph) for ph in np.arange(min_ph, max_ph + ph_spacing, ph_spacing)]


def write_run_manifest(
    output_path: Path,
    *,
    protein: str,
    ligand: str | None,
    cofactor: str | None,
    titratable_residue_indices: list[int],
    titratable_residue_query: str | None,
    phs: list[float],
    n_replicas: int,
    integrator_step_size_ps: float,
    protonation_swap_interval_ps: float,
    cph_config: ConstantpHSettings,
    titratable_residue_labels: list[str] | None = None,
    allowed_variant_indices: dict[int, list[int]] | None = None,
    weights: list[float] | None = None,
    weight_equilibration_done: bool = False,
) -> None:
    """Write the run manifest describing this constant-pH run's configuration."""
    payload = {
        "version": MANIFEST_VERSION,
        "protein": str(Path(protein).resolve()),
        "ligand": str(Path(ligand).resolve()) if ligand else None,
        "cofactor": str(Path(cofactor).resolve()) if cofactor else None,
        "titratable_residue_indices": titratable_residue_indices,
        # Original-structure identifiers (e.g. "GLU 404 A"), aligned index-for-index
        # with titratable_residue_indices, so the run's residues are traceable to the
        # input PDB numbering without re-deriving the topology.
        "titratable_residue_labels": titratable_residue_labels,
        # Per-residue variant masks (topology residue index -> allowed variant
        # indices) for residues kept titratable only for a charge-neutral
        # tautomer flip (e.g. neutral histidine restricted to HID/HIE). JSON
        # object keys are strings; reload converts them back to int. ``None``
        # when no residue is masked.
        "allowed_variant_indices": (
            {str(k): list(v) for k, v in allowed_variant_indices.items()}
            if allowed_variant_indices
            else None
        ),
        "titratable_residue_query": titratable_residue_query,
        "pHs": phs,
        "n_replicas": n_replicas,
        "integrator_step_size_ps": integrator_step_size_ps,
        "protonation_swap_interval_ps": protonation_swap_interval_ps,
        "cph_config_hash": cph_config.hash(),
        "weights": weights,
        "weight_equilibration_done": weight_equilibration_done,
    }
    manifest_path(output_path).write_text(json.dumps(payload, indent=2))


def read_run_manifest(output_path: Path) -> dict[str, Any]:
    """Read and return the run manifest under ``output_path``."""
    path = manifest_path(output_path)
    if not path.exists():
        raise FileNotFoundError(f"Run manifest not found: {path}")
    return json.loads(path.read_text())


def validate_run_manifest(
    manifest: dict[str, Any],
    *,
    protein: str,
    ligand: str | None,
    cofactor: str | None,
    titratable_residue_indices: list[int] | None,
    titratable_residue_query: str | None,
    phs: list[float],
    n_replicas: int,
    integrator_step_size_ps: float,
    protonation_swap_interval_ps: float,
    cph_config: ConstantpHSettings,
) -> None:
    """Raise if the stored manifest disagrees with the current run's parameters."""
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported manifest version {manifest.get('version')!r}; expected {MANIFEST_VERSION}"
        )

    def _check(field: str, expected: Any, *, optional_paths: bool = False) -> None:
        actual = manifest.get(field)
        if optional_paths and expected is not None:
            expected = str(Path(expected).resolve())
        if actual != expected:
            raise ValueError(
                f"Manifest mismatch for {field!r}: "
                f"checkpoint has {actual!r}, current run has {expected!r}"
            )

    _check("protein", protein, optional_paths=True)
    _check("ligand", ligand, optional_paths=True)
    _check("cofactor", cofactor, optional_paths=True)
    _check("pHs", phs)
    _check("n_replicas", n_replicas)
    _check("integrator_step_size_ps", integrator_step_size_ps)
    _check("protonation_swap_interval_ps", protonation_swap_interval_ps)
    _check("cph_config_hash", cph_config.hash())
    _check("titratable_residue_query", titratable_residue_query)

    if titratable_residue_indices is not None:
        _check("titratable_residue_indices", titratable_residue_indices)
    elif "titratable_residue_indices" in manifest:
        logger.info(
            "Using titratable_residue_indices from manifest: "
            f"{manifest['titratable_residue_indices']}"
        )


def update_manifest_weights(
    output_path: Path,
    weights: list[float],
    *,
    weight_equilibration_done: bool = True,
) -> None:
    """Update the stored simulated-tempering weights in the run manifest."""
    data = read_run_manifest(output_path)
    data["weights"] = weights
    data["weight_equilibration_done"] = weight_equilibration_done
    manifest_path(output_path).write_text(json.dumps(data, indent=2))


def write_production_state(
    directory: Path,
    *,
    batches_completed: int,
    next_remd_swap_ps: float,
    results: list[tuple[float, ...]],
) -> None:
    """Write the production progress state (batches, next swap, results) to disk."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "batches_completed": batches_completed,
        "next_remd_swap_ps": next_remd_swap_ps,
        "results": [list(row) for row in results],
    }
    (directory / PRODUCTION_STATE_FILENAME).write_text(json.dumps(payload))


def read_production_state(directory: Path) -> dict[str, Any]:
    """Read production progress state from disk, or return empty defaults."""
    path = directory / PRODUCTION_STATE_FILENAME
    if not path.exists():
        return {
            "batches_completed": 0,
            "next_remd_swap_ps": None,
            "results": [],
        }
    data = json.loads(path.read_text())
    data["results"] = [tuple(row) for row in data.get("results", [])]
    return data
