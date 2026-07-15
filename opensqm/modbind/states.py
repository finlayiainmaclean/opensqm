"""Build and persist the bound and unbound states for ModBinddG.

The bound state is the ligand-protein complex solvated in water with backbone
restraints applied distal to the ligand (so the protein cannot unfold at high
temperature while the ligand is free to escape). The unbound state is the
ligand alone in a water box. Each prepared state is serialised as a CIF
structure, an OpenMM ``System`` XML, and a small JSON metadata file so runs can
resume without re-equilibrating.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from openmm import XmlSerializer, unit
from openmm.app import PDBxFile

from opensqm.cph.run_cph import SystemState
from opensqm.md.equilibrate import EquilibrationSettings, equilibrate
from opensqm.md.prepare import build_complex_forcefield, create_system, prepare_complex
from opensqm.md.restraints import add_distal_restraints

if TYPE_CHECKING:
    from openmm.app.topology import Topology
    from rdkit import Chem

LIGAND_RESNAME = "LIG"


@dataclass(kw_only=True)
class PreparedState(SystemState):
    """An escape-ready :class:`SystemState` plus ligand bookkeeping."""

    ligand_indices: list[int]
    is_bound: bool


def _ligand_heavy_atom_indices(topology: Topology) -> list[int]:
    indices = [
        atom.index
        for atom in topology.atoms()
        if atom.residue.name == LIGAND_RESNAME
        and atom.element is not None
        and atom.element.symbol != "H"
    ]
    if not indices:
        raise ValueError(f"No {LIGAND_RESNAME} heavy atoms found in topology")
    return indices


def build_bound_state_from_state(
    state: SystemState,
) -> PreparedState:
    """Build the restrained bound-state system from a pre-equilibrated snapshot.

    ``state`` is a CpH lowest-energy frame of the solvated ligand-protein
    complex - already equilibrated and carrying its periodic box - so its
    topology/positions are used as-is. Only the OpenMM ``System`` is built (to
    attach the escape restraints); no re-solvation or re-equilibration is done.

    The ligand parameters come from ``state.ligand`` (the LIG residue in the
    exact protonation state of this frame). ``ligand_rdmol`` is a fallback for
    when CpH did not record one.
    """
    if state.ligand is None:
        raise ValueError("build_bound_state_from_state needs a ligand: state.ligand is None ")

    topology = state.topology
    positions = state.positions

    logger.info("Building bound-state system from pre-equilibrated snapshot")
    forcefield = build_complex_forcefield(
        state.ligand,
        solvent_mode="explicit",
    )

    system = create_system(forcefield, topology)
    # Restrain protein backbone distal to the ligand so it stays folded at high
    # temperature while the ligand escapes. Paper: flat-bottom sigma = 3.0 A on
    # backbone atoms, except residues within 6 A of the binding site (left free).
    system, _ = add_distal_restraints(
        system,
        positions,
        topology.atoms(),
        restraints=("backbone",),
        flat_bottom_sigma=3.0,
        exclusion_distance=0.6,
        exclude_by_residue=True,
        min_distance=0.6,
        max_distance=1.0,
        max_restraint_force=100.0,
    )

    return PreparedState(
        topology=topology,
        positions=positions,
        system=system,
        ligand_indices=_ligand_heavy_atom_indices(topology),
        is_bound=True,
    )


def build_unbound_state(
    ligand_rdmol: Chem.Mol,
    *,
    equilibration_config: EquilibrationSettings,
    box_shape: str = "cube",
    padding: unit.Quantity = 1.2 * unit.nanometer,
) -> PreparedState:
    """Solvate and equilibrate the ligand alone in a water box."""
    padding_nm = padding.value_in_unit(unit.nanometer)

    logger.info("Preparing unbound ligand (ligand in water)")
    topology, positions, forcefield = prepare_complex(
        ligand_rdmol,
        padding=padding_nm,
        protein_modeller=None,
        box_shape=box_shape,
        solvent_mode="explicit",
    )

    logger.info("Equilibrating unbound ligand")
    topology, positions = equilibrate(
        topology,
        positions,
        forcefield,
        config=equilibration_config,
    )

    system = create_system(forcefield, topology)

    return PreparedState(
        topology=topology,
        positions=positions,
        system=system,
        ligand_indices=_ligand_heavy_atom_indices(topology),
        is_bound=False,
    )


def _structure_path(directory: Path, name: str) -> Path:
    return directory / f"{name}_structure.cif"


def _system_path(directory: Path, name: str) -> Path:
    return directory / f"{name}_system.xml"


def _meta_path(directory: Path, name: str) -> Path:
    return directory / f"{name}_meta.json"


def save_prepared_state(state: PreparedState, directory: Path, name: str) -> None:
    """Serialise a prepared state to CIF + System XML + JSON metadata."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    with _structure_path(directory, name).open("w") as handle:
        PDBxFile.writeFile(state.topology, state.positions, handle, keepIds=True)

    _system_path(directory, name).write_text(XmlSerializer.serialize(state.system))

    box_vectors = state.topology.getPeriodicBoxVectors()
    box_vectors_nm = None
    if box_vectors is not None:
        box_vectors_nm = [
            [float(component.value_in_unit(unit.nanometer)) for component in row]
            for row in box_vectors
        ]
    meta = {
        "ligand_indices": [int(i) for i in state.ligand_indices],
        "is_bound": bool(state.is_bound),
        "box_vectors_nm": box_vectors_nm,
    }
    _meta_path(directory, name).write_text(json.dumps(meta))
    logger.info(f"Saved equilibrated {name} state to {directory}")


def load_prepared_state(directory: Path, name: str) -> PreparedState | None:
    """Load a prepared state from disk, or return ``None`` if absent."""
    directory = Path(directory)
    structure_path = _structure_path(directory, name)
    system_path = _system_path(directory, name)
    meta_path = _meta_path(directory, name)
    if not (structure_path.exists() and system_path.exists() and meta_path.exists()):
        return None

    cif = PDBxFile(str(structure_path))
    system = XmlSerializer.deserialize(system_path.read_text())
    meta = json.loads(meta_path.read_text())

    topology = cif.topology
    box_vectors_nm = meta.get("box_vectors_nm")
    if box_vectors_nm is not None:
        topology.setPeriodicBoxVectors(box_vectors_nm * unit.nanometer)

    return PreparedState(
        topology=topology,
        positions=cif.positions,
        system=system,
        ligand_indices=list(meta["ligand_indices"]),
        is_bound=bool(meta["is_bound"]),
    )
