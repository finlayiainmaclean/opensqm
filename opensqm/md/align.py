"""Realign MD-derived structures back onto the input protein's frame.

A frame pulled out of a trajectory does not sit where the input protein sat:
``Modeller.addSolvent`` recentres the complex in its periodic box, and
unrestrained MD then lets the whole protein translate and tumble inside that
box. The published representative structures are meant to be read against the
input protein - overlaid on it in a viewer, or rescored alongside it - so the
rigid-body part of that drift is undone here with a Kabsch fit of the protein
C-alpha atoms before the structures are written out.

Only the *written* structures are realigned. A solvated snapshot that MD will be
resumed from must stay in its own frame: rotating the contents of a periodic box
while leaving the box vectors alone breaks the lattice (the periodic images no
longer tile), so :func:`align_positions_to_reference` is applied to the trimmed,
non-periodic outputs only.

The fit is best-effort by design: an unmatchable reference (unreadable PDB,
renumbered residues) is logged and the structure is written in its own frame
rather than failing a run that has already paid for the MD.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from loguru import logger
from openmm import unit
from openmm.app.pdbfile import PDBFile

from opensqm.md.fix import ALL_PROTEIN_RESNAMES

if TYPE_CHECKING:
    from collections.abc import Iterable

    from openmm.app.topology import Topology

# The rigid fit is built from C-alpha atoms: one per protein residue is plenty to
# pin the fold's frame, and it is the one atom name guaranteed to be present in
# both an unprepared input PDB and a prepared, protonated, solvated topology.
_ALIGNMENT_ATOM_NAME = "CA"

# Below this many matched C-alpha pairs a rigid transform is not meaningfully
# determined, so the structure is left in its own frame.
_MIN_ALIGNMENT_ATOMS = 3

# A fit residual above this is reported as a warning: at that point the residue
# correspondence is more likely wrong (renumbered input) than the protein having
# genuinely deformed that much during MD.
_SUSPECT_FIT_RMSD_ANGSTROM = 5.0

_NM_TO_ANGSTROM = 10.0

# A protein residue's PDB identity: (chain id, residue id, insertion code), or
# (residue id, insertion code) once the chain-agnostic fallback has dropped the
# chain id.
_ResidueKey = tuple[str, ...]
_V = TypeVar("_V")


def kabsch_rt(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rigid transform mapping ``mobile`` onto ``target`` (Kabsch).

    Returns ``(R, t)`` such that ``x_aligned = R @ x + t`` best superposes the
    ``mobile`` point set onto ``target`` in a least-squares sense (row-major
    arrays: ``aligned = mobile @ R.T + t``). The reflection-correcting
    determinant sign keeps ``R`` a proper rotation.
    """
    mob_c = mobile.mean(axis=0)
    tgt_c = target.mean(axis=0)
    p = mobile - mob_c
    q = target - tgt_c
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    translation = tgt_c - rotation @ mob_c
    return rotation, translation


def _as_nm(positions: unit.Quantity | np.ndarray) -> np.ndarray:
    """Coordinates as a plain ``(n_atoms, 3)`` array in nm."""
    if isinstance(positions, unit.Quantity):
        return np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
    return np.asarray(positions, dtype=float)


def _calpha_keys(topology: Topology) -> list[tuple[_ResidueKey, int]]:
    """``((chain id, residue id, insertion code), atom index)`` per protein C-alpha.

    Keyed by the residue's PDB identity rather than by its position in the
    topology, so the input protein and the prepared complex match up despite
    preparation reordering atoms (the ligand goes first), adding hydrogens and
    missing residues, and appending waters/ions. Residue *names* are deliberately
    left out of the key: PROPKA/PDB2PQR can rename a titratable residue
    (HIS -> HID/HIE/HIP, ...) without moving it. Restricting to
    :data:`ALL_PROTEIN_RESNAMES` keeps a calcium ion - residue ``CA``, atom
    ``CA`` - out of the backbone fit.
    """
    keys: list[tuple[_ResidueKey, int]] = []
    for residue in topology.residues():
        if residue.name not in ALL_PROTEIN_RESNAMES:
            continue
        for atom in residue.atoms():
            if atom.name == _ALIGNMENT_ATOM_NAME:
                key = (
                    str(residue.chain.id or ""),
                    str(residue.id),
                    str(residue.insertionCode or ""),
                )
                keys.append((key, atom.index))
                break
    return keys


def _unique(pairs: Iterable[tuple[_ResidueKey, _V]]) -> dict[_ResidueKey, _V]:
    """Index the ``(key, value)`` pairs, dropping any key that is not unique.

    An ambiguous key (duplicate residue numbering within a chain) would pair
    arbitrary residues, so it is excluded from the fit rather than guessed at.
    """
    indexed: dict[_ResidueKey, _V] = {}
    duplicated: set[_ResidueKey] = set()
    for key, value in pairs:
        if key in indexed:
            duplicated.add(key)
        else:
            indexed[key] = value
    for key in duplicated:
        del indexed[key]
    return indexed


def _drop_chain(keyed: dict[_ResidueKey, _V]) -> dict[_ResidueKey, _V]:
    """Re-key on residue id + insertion code alone, dropping newly ambiguous keys."""
    return _unique((key[1:], value) for key, value in keyed.items())


def _reference_calphas(reference_pdb: Path | str) -> dict[_ResidueKey, np.ndarray] | None:
    """C-alpha coordinates (nm) of ``reference_pdb``, keyed by residue identity."""
    try:
        pdb = PDBFile(str(reference_pdb))
    except Exception as exc:
        # Best-effort by design: never fail a run that has already paid for its MD
        # just because the reference could not be parsed.
        logger.warning(f"Could not read {reference_pdb} as an alignment reference: {exc}")
        return None
    coords = _as_nm(pdb.positions)
    return {key: coords[index] for key, index in _unique(_calpha_keys(pdb.topology)).items()}


def _correspondence(
    topology: Topology, reference: dict[_ResidueKey, np.ndarray]
) -> tuple[list[int], np.ndarray]:
    """Pair up the C-alphas: ``(mobile atom indices, reference coordinates nm)``."""
    mobile = _unique(_calpha_keys(topology))
    shared = sorted(key for key in mobile if key in reference)
    if len(shared) < _MIN_ALIGNMENT_ATOMS:
        # Chain ids are not always preserved end to end (and are sometimes absent
        # from the input PDB entirely), so fall back to residue numbering alone.
        mobile, reference = _drop_chain(mobile), _drop_chain(reference)
        shared_nc = sorted(key for key in mobile if key in reference)
        if len(shared_nc) > len(shared):
            logger.info(
                f"Matched only {len(shared)} C-alpha atom(s) on chain id + residue "
                f"number; falling back to chain-agnostic matching ({len(shared_nc)} atoms)"
            )
        shared = shared_nc
    if not shared:
        return [], np.empty((0, 3))
    return [mobile[key] for key in shared], np.array([reference[key] for key in shared])


def align_positions_to_reference(
    topology: Topology,
    positions: unit.Quantity | np.ndarray,
    reference_pdb: Path | str,
    *,
    label: str = "frame",
) -> np.ndarray:
    """Rigidly map ``positions`` onto ``reference_pdb``'s frame; return them in nm.

    ``topology`` describes ``positions`` (any mix of protein, ligand, waters and
    ions); the transform is fitted on the protein C-alpha atoms shared with
    ``reference_pdb`` and then applied to *every* atom, so the complex moves as
    one rigid body and no internal geometry - or interatomic distance - changes.

    Returns the realigned coordinates (n_atoms x 3, nm), or the input
    coordinates unchanged when the reference cannot be read or too few residues
    match; either way the caller can write the result without special-casing.
    """
    coords = _as_nm(positions)
    reference = _reference_calphas(reference_pdb)
    if reference is None:
        return coords

    indices, reference_coords = _correspondence(topology, reference)
    if len(indices) < _MIN_ALIGNMENT_ATOMS:
        logger.warning(
            f"Only {len(indices)} C-alpha atom(s) of the {label} matched "
            f"{Path(str(reference_pdb)).name}; leaving it unaligned (its own MD frame)"
        )
        return coords

    rotation, translation = kabsch_rt(coords[indices], reference_coords)
    aligned = coords @ rotation.T + translation

    fit_rmsd = (
        float(np.sqrt(((aligned[indices] - reference_coords) ** 2).sum(axis=1).mean()))
        * _NM_TO_ANGSTROM
    )
    shift = (
        float(np.linalg.norm(coords[indices].mean(axis=0) - reference_coords.mean(axis=0)))
        * _NM_TO_ANGSTROM
    )
    angle = math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))
    logger.info(
        f"Realigned the {label} onto {Path(str(reference_pdb)).name} over "
        f"{len(indices)} C-alpha atom(s): undid a {shift:.1f} A translation and a "
        f"{angle:.1f} deg rotation, C-alpha RMSD after the fit {fit_rmsd:.2f} A"
    )
    if fit_rmsd > _SUSPECT_FIT_RMSD_ANGSTROM:
        logger.warning(
            f"C-alpha RMSD after realigning the {label} is {fit_rmsd:.2f} A - the "
            "residue correspondence with the input protein may be wrong, or the fold "
            "moved substantially during MD"
        )
    return aligned
