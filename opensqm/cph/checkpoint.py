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
    return output_path / MANIFEST_FILENAME


def checkpoint_dir(output_path: Path) -> Path:
    return output_path / CHECKPOINT_DIRNAME


def equilibrated_pdb_path(output_path: Path) -> Path:
    return output_path / EQUILIBRATED_PDB


def pH_ladder(min_pH: float, max_pH: float, pH_spacing: float) -> list[float]:
    import numpy as np

    return [float(pH) for pH in np.arange(min_pH, max_pH + pH_spacing, pH_spacing)]


def write_run_manifest(
    output_path: Path,
    *,
    protein: str,
    ligand: str | None,
    cofactor: str | None,
    titratable_residue_indices: list[int],
    titratable_residue_query: str | None,
    pHs: list[float],
    n_replicas: int,
    integrator_step_size_ps: float,
    protonation_swap_interval_ps: float,
    cph_config: ConstantpHSettings,
    weights: list[float] | None = None,
    weight_equilibration_done: bool = False,
) -> None:
    payload = {
        "version": MANIFEST_VERSION,
        "protein": str(Path(protein).resolve()),
        "ligand": str(Path(ligand).resolve()) if ligand else None,
        "cofactor": str(Path(cofactor).resolve()) if cofactor else None,
        "titratable_residue_indices": titratable_residue_indices,
        "titratable_residue_query": titratable_residue_query,
        "pHs": pHs,
        "n_replicas": n_replicas,
        "integrator_step_size_ps": integrator_step_size_ps,
        "protonation_swap_interval_ps": protonation_swap_interval_ps,
        "cph_config_hash": cph_config.hash(),
        "weights": weights,
        "weight_equilibration_done": weight_equilibration_done,
    }
    manifest_path(output_path).write_text(json.dumps(payload, indent=2))


def read_run_manifest(output_path: Path) -> dict[str, Any]:
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
    pHs: list[float],
    n_replicas: int,
    integrator_step_size_ps: float,
    protonation_swap_interval_ps: float,
    cph_config: ConstantpHSettings,
) -> None:
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported manifest version {manifest.get('version')!r}; "
            f"expected {MANIFEST_VERSION}"
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
    _check("pHs", pHs)
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
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "batches_completed": batches_completed,
        "next_remd_swap_ps": next_remd_swap_ps,
        "results": [list(row) for row in results],
    }
    (directory / PRODUCTION_STATE_FILENAME).write_text(json.dumps(payload))


def read_production_state(directory: Path) -> dict[str, Any]:
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
