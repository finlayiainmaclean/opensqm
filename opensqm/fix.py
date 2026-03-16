from pathlib import Path

import numpy as np
from openmm import unit
from openmm.app import Atom, Modeller, PDBFile, Residue, Topology
from pdbfixer import PDBFixer

# Max allowed distance (nm) before we consider the chain broken
BREAK_THRESHOLD_NM = 0.25  # ~2.5 Å — generous but catches true breaks


def _is_protein(res: Residue) -> bool:
    atom_names = {a.name for a in res.atoms()}
    return "CA" in atom_names


def _get_atom_by_name(res: Residue, name: str):
    return next((a for a in res.atoms() if a.name == name), None)


def _distance_nm(pos, a1, a2) -> float:
    p1 = pos[a1.index].value_in_unit(unit.nanometer)
    p2 = pos[a2.index].value_in_unit(unit.nanometer)
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _is_chain_break(
    prev_res: Residue,
    curr_res: Residue,
    positions,
) -> bool:
    """
    Return True if the bond connecting prev_res → curr_res is missing or
    longer than a threshold nanometres.
    """
    tail_atom = _get_atom_by_name(prev_res, "C")
    head_atom = _get_atom_by_name(curr_res, "N")

    # Missing connector atom → treat as a break
    if tail_atom is None or head_atom is None:
        return True

    return _distance_nm(positions, tail_atom, head_atom) > BREAK_THRESHOLD_NM


def renumber_chains(fixer: PDBFixer) -> PDBFixer:
    """
    Split chains wherever the geometric distance between consecutive residues'
    connector atoms exceeds BREAK_THRESHOLD_NM, regardless of residue IDs.
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


def run_pdbfixer(
    input_protein_path: Path,
    output_protein_path: Path,
    keep_waters: bool = True,
    keep_ions: bool = True,
    pH: float = 7.4,
):

    input_protein_path = Path(input_protein_path)
    output_protein_path = Path(output_protein_path)

    fixer = PDBFixer(filename=str(input_protein_path))

    if keep_ions:
        # Extract ion atoms before any modifications
        ion_atoms = []
        ion_positions = []

        for residue in fixer.topology.residues():
            if residue.name in ["ZN", "MG", "CA", "FE", "CU", "MN", "CO", "NA", "K", "NI", "MO"]:
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
    fixer.missingTerminals = {}
    fixer.addMissingAtoms()

    # fixer1 = renumber_chains(fixer)
    # fixer.topology, fixer.positions = fixer1.topology, fixer1.positions

    # fixer.addCaps(fixer)

    # residues = list(fixer.topology.residues())
    # residues_to_remove = [residues[0]]
    # fixer.topology, fixer.positions = crop(fixer.topology, fixer.positions, residues_to_remove)
    fixer.addMissingHydrogens(pH)

    # Add ion atoms back to the structure
    if keep_ions and ion_atoms:
        # Create a modeller to add the ion atoms

        modeller = Modeller(fixer.topology, fixer.positions)

        # Create ion topology and positions
        for ion_atom, ion_pos in zip(ion_atoms, ion_positions, strict=False):
            # Create a new residue and chain for each ion atom
            ion_final_positions = []
            ion_topology = Topology()
            ion_chain = ion_topology.addChain()
            ion_residue = ion_topology.addResidue(ion_atom["name"], ion_chain)
            ion_topology.addAtom(ion_atom["name"], ion_atom["element"], ion_residue)
            ion_final_positions.append(ion_pos)

            # Add ion atom to the modeller
            modeller.add(ion_topology, ion_final_positions)

        # Update fixer with the new topology and positions
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions

    # fixer = flip_residues(fixer)

    PDBFile.writeFile(
        fixer.topology, fixer.positions, open(str(output_protein_path), "w"), keepIds=True
    )
    return fixer


if __name__ == "__main__":
    run_pdbfixer("/tmp/7RPZ.pdb", "/tmp/prot.pdb", keep_waters=False)
