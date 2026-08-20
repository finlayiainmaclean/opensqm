"""PDBFixer-based preparation of protein structures (add hydrogens, renumber chains)."""

import itertools
import logging
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click
import numpy as np
from loguru import logger
from openmm import Vec3, unit
from openmm.app import Atom, Modeller, PDBFile, Residue, Topology, element
from pdb2pqr.main import run_pdb2pqr
from pdbfixer import PDBFixer

from opensqm.md.terminal_ring_mc import (
    RING_FLIP_BONDS,
    find_residue_ring_bond,
    find_terminal_group,
    rotate_terminal_group,
)

# Max allowed distance (nm) before we consider the chain broken
BREAK_THRESHOLD_NM = 0.25  # ~2.5 Å — generous but catches true breaks

# Monatomic ions to preserve across preparation (PDBFixer.removeHeterogens strips
# these, so they are extracted up front and re-added after protonation).
ION_RESNAMES = ("ZN", "MG", "CA", "FE", "CU", "MN", "CO", "NA", "K", "NI", "MO")

# PROPKA/PDB2PQR log the full titration curve at INFO; keep only warnings/errors.
for _name in ("pdb2pqr", "propka"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class PDB2PQRError(Exception):
    """PDB2PQR/PROPKA failed to protonate a structure."""


def _is_protein(res: Residue) -> bool:
    atom_names = {a.name for a in res.atoms()}
    return "CA" in atom_names


def _get_atom_by_name(res: Residue, name: str) -> Atom | None:
    return next((a for a in res.atoms() if a.name == name), None)


def _distance_nm(pos: unit.Quantity, a1: Atom, a2: Atom) -> float:
    p1 = pos[a1.index].value_in_unit(unit.nanometer)
    p2 = pos[a2.index].value_in_unit(unit.nanometer)
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _is_chain_break(
    prev_res: Residue,
    curr_res: Residue,
    positions: unit.Quantity,
) -> bool:
    """
    Return True if the bond connecting prev_res → curr_res is missing or too long.

    The threshold is BREAK_THRESHOLD_NM (nanometres).
    """
    tail_atom = _get_atom_by_name(prev_res, "C")
    head_atom = _get_atom_by_name(curr_res, "N")

    # Missing connector atom → treat as a break
    if tail_atom is None or head_atom is None:
        return True

    return _distance_nm(positions, tail_atom, head_atom) > BREAK_THRESHOLD_NM


def renumber_chains(fixer: PDBFixer) -> PDBFixer:
    """
    Split chains at breaks (large gaps between consecutive residues).

    Chains are split wherever the distance between consecutive residues' connector
    atoms exceeds BREAK_THRESHOLD_NM, regardless of residue IDs.
    """
    positions = fixer.positions
    old_top = fixer.topology
    new_top = Topology()
    atom_map: dict[Atom, Atom] = {}

    for chain in old_top.chains():
        residues = list(chain.residues())
        new_chain = new_top.addChain(chain.id)

        for i, res in enumerate(residues):
            if i > 0 and _is_chain_break(residues[i - 1], res, positions):
                new_chain = new_top.addChain(chain.id)

            new_res = new_top.addResidue(res.name, new_chain, res.id, res.insertionCode)
            for atom in res.atoms():
                atom_map[atom] = new_top.addAtom(atom.name, atom.element, new_res)

    # Preserve only intra-chain bonds
    for bond in old_top.bonds():
        new_a1 = atom_map[bond[0]]
        new_a2 = atom_map[bond[1]]
        if new_a1.residue.chain == new_a2.residue.chain:
            new_top.addBond(new_a1, new_a2)

    fixer.topology = new_top
    fixer.positions = positions
    return fixer


def _protonate_with_propka(input_pdb: Path, output_pdb: Path, ph: float) -> list[dict]:
    """Assign titration states and optimise the H-bond network with PROPKA/PDB2PQR.

    Runs PDB2PQR (which drives PROPKA internally) to, at the requested ``ph``:

      1. compute empirical pKa values for each titratable residue's local
         environment (PROPKA);
      2. assign titration states from those pKa values;
      3. flip the side chains of HIS, ASN, and GLN;
      4. rotate the sidechain hydrogen on SER, THR, TYR, and CYS (where present);
      5. place the sidechain hydrogen on neutral HIS and protonated GLU/ASP; and
      6. optimise all water hydrogens.

    ``--ff=AMBER`` selects the parameter set used for the optimisation, while the
    written PDB keeps canonical residue names (HIS/ASP/GLU/CYS/...) — the chosen
    protonation state is encoded by which hydrogens are present (e.g. HD1 vs HE2
    on HIS). The PQR output is required by PDB2PQR but discarded here.

    Returns PDB2PQR's own per-group PROPKA pKa predictions -- one ``dict`` per
    ionisable group (keys include ``res_name``/``res_num``/``chain_id``/
    ``pKa``) from the exact same PROPKA pass that decided the titration
    states above, so a caller needing pKa *values* (not just the resulting
    structure) doesn't need to run PROPKA a second time.
    """
    with tempfile.TemporaryDirectory() as tmp:
        pqr_path = Path(tmp) / "structure.pqr"
        try:
            _missed_residues, pka_groups, _biomolecule = run_pdb2pqr(
                [
                    "--ff=AMBER",
                    "--keep-chain",
                    "--titration-state-method=propka",
                    f"--with-ph={ph}",
                    f"--pdb-output={output_pdb}",
                    str(input_pdb),
                    str(pqr_path),
                ]
            )
        except RuntimeError as err:
            # PDB2PQR logs the real reason at CRITICAL then re-raises a bare, empty
            # `RuntimeError` — the message survives only on __cause__, so without this
            # the pod failure reaches Temporal with an empty error summary.
            raise PDB2PQRError(f"PDB2PQR failed: {err.__cause__ or err}") from err
    return pka_groups or []


def _his_tautomer(residue: Residue) -> str | None:
    """Return the HIS tautomer/protonation state present on ``residue``.

    Inferred from which ring-nitrogen hydrogens are present (see
    :func:`_protonate_with_propka`): ``"HID"`` (HD1 only), ``"HIE"`` (HE2 only),
    ``"HIP"`` (both, doubly protonated/charged), or ``None`` if neither is
    present (not a valid HIS state; should not occur on a PROPKA-protonated
    structure).
    """
    names = {a.name for a in residue.atoms()}
    hd1, he2 = "HD1" in names, "HE2" in names
    if hd1 and he2:
        return "HIP"
    if hd1:
        return "HID"
    if he2:
        return "HIE"
    return None


@dataclass(frozen=True)
class _FlipCandidate:
    """One independently-toggleable HIS/ASN/GLN flip near the ligand.

    ``"tautomer"`` moves a neutral HIS's ring-nitrogen hydrogen to the other
    nitrogen (HID<->HIE); ``"ring"`` rigidly rotates the residue's terminal
    ring/amide group 180 degrees about its
    :data:`opensqm.md.terminal_ring_mc.RING_FLIP_BONDS` bond. These are
    orthogonal degrees of freedom, so a single near-ligand HIS residue can
    contribute both a ``"tautomer"`` and a ``"ring"`` candidate.
    """

    residue_index: int
    label: str
    kind: Literal["tautomer", "ring"]
    distance_angstrom: float


def _nearest_atom_distance(
    residue: Residue, pos_ang: np.ndarray, ref_coords_angstrom: np.ndarray
) -> float:
    atom_indices = [a.index for a in residue.atoms()]
    return float(
        np.linalg.norm(
            ref_coords_angstrom[:, None, :] - pos_ang[None, atom_indices, :], axis=-1
        ).min()
    )


def find_flippable_residues(
    topology: Topology,
    positions: unit.Quantity,
    ref_coords_angstrom: np.ndarray,
    cutoff_angstrom: float = 5.0,
) -> list[_FlipCandidate]:
    """List HIS/ASN/GLN flip candidates within ``cutoff_angstrom`` of ``ref_coords_angstrom``.

    PROPKA/PDB2PQR's H-bond network optimisation (:func:`_protonate_with_propka`)
    and PDBFixer's heavy-atom placement only ever see the apo protein, so a
    residue near the ligand may have been assigned a state that satisfies its
    *other* protein neighbours rather than the ligand. This finds candidates
    for :func:`enumerate_residue_flip_variants` to try the alternative state
    on -- see :class:`_FlipCandidate` for the two kinds of flip and why HIS
    can contribute both.

    Returned nearest-first by distance to ``ref_coords_angstrom`` (typically
    the ligand's atom positions, Angstrom), so a caller that must truncate the
    list (see :func:`enumerate_residue_flip_variants`'s ``max_variants``)
    keeps the candidates most likely to actually matter.
    """
    pos_ang = np.array([p.value_in_unit(unit.angstrom) for p in positions], dtype=float)
    candidates: list[_FlipCandidate] = []
    for index, residue in enumerate(topology.residues()):
        resname = residue.name
        if resname != "HIS" and resname not in RING_FLIP_BONDS:
            continue
        distance = _nearest_atom_distance(residue, pos_ang, ref_coords_angstrom)
        if distance >= cutoff_angstrom:
            continue
        label = f"{residue.chain.id}/{resname}{residue.id}"
        if resname == "HIS" and _his_tautomer(residue) in ("HID", "HIE"):
            candidates.append(_FlipCandidate(index, label, "tautomer", distance))
        if resname in RING_FLIP_BONDS:
            candidates.append(_FlipCandidate(index, label, "ring", distance))
    candidates.sort(key=lambda c: c.distance_angstrom)
    return candidates


def enumerate_residue_flip_variants(
    topology: Topology,
    positions: unit.Quantity,
    candidates: list[_FlipCandidate],
    max_variants: int = 8,
) -> list[tuple[str, Topology, unit.Quantity]]:
    """Enumerate every combination of ``candidates``' flips.

    Each combination applies, in order:

    1. every chosen ``"tautomer"`` candidate's HID<->HIE swap, realised with a
       single ``Modeller.addHydrogens(variants=...)`` call passing ``None``
       for every other residue so every residue PROPKA already assigned
       (ASH/GLH/HIP/LYN/CYX/...) keeps its exact existing protonation state --
       skipped entirely when no ``"tautomer"`` candidate is chosen; then
    2. every chosen ``"ring"`` candidate's 180-degree rigid rotation about its
       :data:`opensqm.md.terminal_ring_mc.RING_FLIP_BONDS` bond, applied via
       :func:`opensqm.md.terminal_ring_mc.rotate_terminal_group` (each
       candidate's rotatable atoms are disjoint from every other's, so
       applying them one at a time is order-independent).

    Returns ``(label, topology, positions)`` triples, always including the
    unflipped baseline first (``label=""``, matching the input structure);
    ``label`` comma-joins every candidate applied in that combination (e.g.
    ``"A/HIS208:HIE->HID,A/HIS208:ring-flip,A/ASN45:ring-flip"``).

    ``candidates`` longer than ``floor(log2(max_variants))`` is truncated to
    its first that many entries (logged, not silent) to bound the ``2**n``
    combinatorial blow-up -- pass them ordered by priority (e.g.
    :func:`find_flippable_residues`'s nearest-first order).
    """
    if not candidates:
        return [("", topology, positions)]

    max_candidates = max(1, math.floor(math.log2(max_variants)))
    if len(candidates) > max_candidates:
        dropped = [f"{c.label}:{c.kind}" for c in candidates[max_candidates:]]
        logger.warning(
            f"{len(candidates)} HIS/ASN/GLN flip candidates qualify; keeping "
            f"only the {max_candidates} closest to bound the 2^n combinatorial "
            f"blow-up (dropping {dropped})"
        )
        candidates = candidates[:max_candidates]

    residues = list(topology.residues())
    flip_to = {"HID": "HIE", "HIE": "HID"}
    n_res = topology.getNumResidues()

    variants_out: list[tuple[str, Topology, unit.Quantity]] = []
    for combo in itertools.product((False, True), repeat=len(candidates)):
        chosen = [c for c, do_flip in zip(candidates, combo, strict=True) if do_flip]
        tautomer_flips = [c for c in chosen if c.kind == "tautomer"]
        ring_flips = [c for c in chosen if c.kind == "ring"]

        applied_labels: list[str] = []
        if tautomer_flips:
            variants: list[str | None] = [None] * n_res
            for candidate in tautomer_flips:
                current_state = _his_tautomer(residues[candidate.residue_index])
                target_state = flip_to[current_state]
                variants[candidate.residue_index] = target_state
                applied_labels.append(f"{candidate.label}:{current_state}->{target_state}")
            modeller = Modeller(topology, positions)
            modeller.addHydrogens(variants=variants)
            out_topology, out_positions = modeller.topology, modeller.positions
        else:
            out_topology, out_positions = topology, positions

        if ring_flips:
            pos_nm = np.array([p.value_in_unit(unit.nanometer) for p in out_positions], dtype=float)
            out_residues = list(out_topology.residues())
            for candidate in ring_flips:
                residue = out_residues[candidate.residue_index]
                anchor_idx, pivot_idx = find_residue_ring_bond(
                    out_topology, residue.name, residue.id, residue.chain.id
                )
                group = find_terminal_group(out_topology, anchor_idx, pivot_idx)
                pos_nm = rotate_terminal_group(
                    pos_nm, group.bond[0], group.bond[1], group.rotatable_group, 180.0
                )
                applied_labels.append(f"{candidate.label}:ring-flip")
            out_positions = unit.Quantity([Vec3(*row) for row in pos_nm], unit.nanometer)

        variants_out.append((",".join(applied_labels), out_topology, out_positions))
    return variants_out


def run_pdbfixer(
    input_protein_path: Path,
    output_protein_path: Path,
    keep_waters: bool = True,
    keep_ions: bool = True,
    ph: float = 7.0,
) -> tuple[Path, list[dict]]:
    """Prepare a protein structure and write the protonated result.

    PDBFixer completes the structure (missing residues/atoms, standard residue
    substitution), then PROPKA/PDB2PQR assigns pH-dependent titration states and
    optimises the hydrogen-bonding network — see :func:`_protonate_with_propka`.
    Waters are protonated and optimised in place; monatomic ions are stripped by
    ``removeHeterogens`` and re-added afterwards.

    Returns ``(output_path, pka_groups)`` -- ``pka_groups`` is
    :func:`_protonate_with_propka`'s own per-group PROPKA pKa predictions
    (see its docstring), from the same PROPKA pass that decided the
    titration states written to ``output_path``, so a caller needing pKa
    values doesn't need a second, separate PROPKA run.
    """
    input_protein_path = Path(input_protein_path)
    output_protein_path = Path(output_protein_path)

    fixer = PDBFixer(filename=str(input_protein_path))

    # Extract ion atoms before any modifications; removeHeterogens strips them.
    ion_atoms: list[dict] = []
    ion_positions: list[unit.Quantity] = []
    if keep_ions:
        for residue in fixer.topology.residues():
            if residue.name in ION_RESNAMES:
                for atom in residue.atoms():
                    ion_atoms.append(
                        {
                            "name": atom.name,
                            "element": atom.element,
                            "residue_name": residue.name,
                            "residue_id": residue.id,
                            "chain_id": residue.chain.id,
                        }
                    )
                    ion_positions.append(fixer.positions[atom.index])

    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=keep_waters)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Hydrogens are added by PROPKA/PDB2PQR (pKa-informed), not PDBFixer: write the
    # completed heavy-atom structure, protonate it, then read the result back.
    # Hydrogens already on the input are dropped first, so re-preparing an
    # already-prepared PDB works: PDB2PQR bonds atoms through its own residue
    # templates, and an H it did not place itself (e.g. HD2 on a protonated ASP,
    # which lives only in its ASH patch) is left bondless and aborts debumping with
    # "Found gap in biomolecule structure".
    heavy = Modeller(fixer.topology, fixer.positions)
    heavy.delete([a for a in heavy.topology.atoms() if a.element == element.hydrogen])
    with tempfile.TemporaryDirectory() as tmp:
        heavy_pdb = Path(tmp) / "heavy.pdb"
        protonated_pdb = Path(tmp) / "protonated.pdb"
        with heavy_pdb.open("w") as handle:
            PDBFile.writeFile(heavy.topology, heavy.positions, handle, keepIds=True)
        pka_groups = _protonate_with_propka(heavy_pdb, protonated_pdb, ph)
        protonated = PDBFile(str(protonated_pdb))
        topology, positions = protonated.topology, protonated.positions

    # Add the ion atoms back to the structure.
    if keep_ions and ion_atoms:
        modeller = Modeller(topology, positions)
        for ion_atom, ion_pos in zip(ion_atoms, ion_positions, strict=False):
            # Create a new residue and chain for each ion atom.
            ion_topology = Topology()
            ion_chain = ion_topology.addChain()
            ion_residue = ion_topology.addResidue(ion_atom["name"], ion_chain)
            ion_topology.addAtom(ion_atom["name"], ion_atom["element"], ion_residue)
            modeller.add(ion_topology, [ion_pos])
        topology, positions = modeller.topology, modeller.positions

    with output_protein_path.open("w") as handle:
        PDBFile.writeFile(topology, positions, handle, keepIds=True)
    return output_protein_path, pka_groups


@click.command()
@click.argument("input_protein_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_protein_path", type=click.Path(path_type=Path))
@click.option("--keep-waters", is_flag=True, help="Keep water molecules.")
@click.option("--keep-ions/--no-keep-ions", default=True, help="Keep ion molecules.")
@click.option("--ph", type=float, default=7.4, help="pH for PROPKA titration-state assignment.")
def main(
    input_protein_path: Path,
    output_protein_path: Path,
    keep_waters: bool,
    keep_ions: bool,
    ph: float,
) -> None:
    """Run PDBFixer to prepare protein structures."""
    run_pdbfixer(
        input_protein_path,
        output_protein_path,
        keep_waters=keep_waters,
        keep_ions=keep_ions,
        ph=ph,
    )


if __name__ == "__main__":
    main()
