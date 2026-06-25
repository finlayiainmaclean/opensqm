"""Run-local cache for parameterised ligand OpenFF molecules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openff.toolkit.topology import Molecule  # type: ignore

if TYPE_CHECKING:
    from pathlib import Path

LIGAND_VARIANTS_DIRNAME = "ligand_variants"


def _write_molecule(path: Path, molecule: Molecule) -> None:
    molecule.to_file(str(path), "SDF")


def _read_molecule(path: Path) -> Molecule:
    return Molecule.from_file(str(path))


def ligand_variant_path(output_path: Path, residue_name: str) -> Path:
    return output_path / LIGAND_VARIANTS_DIRNAME / f"{residue_name}.sdf"


def save_ligand_variant(output_path: Path, molecule: Molecule) -> Path:
    """Persist a parameterised OpenFF molecule for a specific run."""
    path = ligand_variant_path(output_path, molecule.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_molecule(path, molecule)
    return path


def load_ligand_variant(output_path: Path, residue_name: str) -> Molecule | None:
    """Load a run-local cached ligand variant, if present."""
    path = ligand_variant_path(output_path, residue_name)
    if not path.exists():
        return None
    return _read_molecule(path)


def persist_ligand_setups(output_path: Path, *setups) -> None:
    """Write all variant molecules (with charges/conformers) under ``output_path``."""
    for setup in setups:
        if setup is None:
            continue
        for molecule in setup.variant_molecules:
            if molecule.partial_charges is None:
                continue
            save_ligand_variant(output_path, molecule)
