"""PDBFixer-based preparation of protein structures (add hydrogens, renumber chains)."""

import logging
import tempfile
from pathlib import Path

import click
import numpy as np
from openmm import unit
from openmm.app import Atom, Modeller, PDBFile, Residue, Topology
from pdb2pqr.main import run_pdb2pqr
from pdbfixer import PDBFixer

# Max allowed distance (nm) before we consider the chain broken
BREAK_THRESHOLD_NM = 0.25  # ~2.5 Å — generous but catches true breaks

# Monatomic ions to preserve across preparation (PDBFixer.removeHeterogens strips
# these, so they are extracted up front and re-added after protonation).
ION_RESNAMES = ("ZN", "MG", "CA", "FE", "CU", "MN", "CO", "NA", "K", "NI", "MO")

# PROPKA/PDB2PQR log the full titration curve at INFO; keep only warnings/errors.
for _name in ("pdb2pqr", "propka"):
    logging.getLogger(_name).setLevel(logging.WARNING)


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


def _protonate_with_propka(input_pdb: Path, output_pdb: Path, ph: float) -> None:
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
    """
    with tempfile.TemporaryDirectory() as tmp:
        pqr_path = Path(tmp) / "structure.pqr"
        run_pdb2pqr(
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


def run_pdbfixer(
    input_protein_path: Path,
    output_protein_path: Path,
    keep_waters: bool = True,
    keep_ions: bool = True,
    ph: float = 7.0,
) -> Path:
    """Prepare a protein structure and write the protonated result.

    PDBFixer completes the structure (missing residues/atoms, standard residue
    substitution), then PROPKA/PDB2PQR assigns pH-dependent titration states and
    optimises the hydrogen-bonding network — see :func:`_protonate_with_propka`.
    Waters are protonated and optimised in place; monatomic ions are stripped by
    ``removeHeterogens`` and re-added afterwards. Returns the output path.
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
    with tempfile.TemporaryDirectory() as tmp:
        heavy_pdb = Path(tmp) / "heavy.pdb"
        protonated_pdb = Path(tmp) / "protonated.pdb"
        with heavy_pdb.open("w") as handle:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
        _protonate_with_propka(heavy_pdb, protonated_pdb, ph)
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
    return output_protein_path


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
