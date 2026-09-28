"""The MMGBSA frame builder matches atoms across OpenMM and mdtraj readings of one PDB."""

import mdtraj as md
import numpy as np
from openmm import unit
from openmm.app import Element, PDBFile, Topology

from opensqm.md.mmgbsa import _openmm_atom_lookup_key


def test_residues_split_by_an_insertion_code_keep_distinct_keys(tmp_path) -> None:
    top = Topology()
    chain = top.addChain("H")
    for code in (" ", "A"):  # chymotrypsin numbering, as in thrombin: 129 then 129A
        residue = top.addResidue("GLY", chain, id="129", insertionCode=code)
        for name, symbol in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")):
            top.addAtom(name, Element.getBySymbol(symbol), residue)
    pdb = tmp_path / "complex.pdb"
    positions = np.arange(top.getNumAtoms() * 3, dtype=float).reshape(-1, 3) * unit.angstrom
    with pdb.open("w") as handle:
        PDBFile.writeFile(top, positions, handle, keepIds=True)

    omm_atoms = list(PDBFile(str(pdb)).topology.atoms())
    keys = [_openmm_atom_lookup_key(a) for a in omm_atoms]
    assert len(set(keys)) == len(keys)

    # get_interaction_energy indexes mdtraj's reading of the file by OpenMM's atom index.
    traj_atoms = list(md.load(str(pdb)).topology.atoms)
    assert [a.name for a in traj_atoms] == [a.name for a in omm_atoms]
