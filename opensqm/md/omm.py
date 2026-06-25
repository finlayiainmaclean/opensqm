"""Convert RDKit molecules to OpenMM topologies and back."""

from loguru import logger
from openmm.app import Atom, Modeller, Topology
from rdkit import Chem


def get_rdkit_pdb_info(atom: Chem.Atom) -> tuple[str, str, int, str] | None:
    """Get PDB info string for RDKit atom."""
    pdb_info = atom.GetPDBResidueInfo()
    if pdb_info is None:
        return None
    chain = pdb_info.GetChainId().strip()
    res_name = pdb_info.GetResidueName().strip()
    res_num = pdb_info.GetResidueNumber()
    atom_name = pdb_info.GetName().strip()
    return chain, res_name, res_num, atom_name


def pdb_residue_key_from_rdkit(atom: Chem.Atom) -> tuple[str, str, int, str] | None:
    """Return a PDB residue key (chain, resname, resnum, icode) for an RDKit atom."""
    pdb_info = atom.GetPDBResidueInfo()
    if pdb_info is None:
        return None
    icode = (pdb_info.GetInsertionCode() or " ").strip() or " "
    return (
        pdb_info.GetChainId().strip(),
        pdb_info.GetResidueName().strip(),
        pdb_info.GetResidueNumber(),
        icode,
    )


def pdb_residue_key_from_openmm(residue) -> tuple[str, str, int, str]:
    """Return a PDB residue key (chain, resname, resnum, icode) for an OpenMM residue."""
    icode = (getattr(residue, "insertionCode", "") or " ").strip() or " "
    try:
        resnum = int(residue.id)
    except (TypeError, ValueError):
        resnum = int("".join(character for character in str(residue.id) if character.isdigit()) or 0)
    return (
        str(residue.chain.id).strip(),
        residue.name.strip(),
        resnum,
        icode,
    )


def map_pdb_residue_keys_to_openmm_indices(
    topology: Topology,
    residue_keys: set[tuple[str, str, int, str]],
) -> set[int]:
    """Map PDB residue keys onto OpenMM residue indices.

    OpenFF-built ligand topologies often leave ``chain.id`` empty in memory
    even though :class:`openmm.app.PDBFile` writes them with a concrete chain
    letter (typically ``A``). When matching RDSL selections from a written
    PDB, treat blank chains as wildcards so ligands are not dropped.
    """
    omm_keys = [
        (pdb_residue_key_from_openmm(residue), residue.index)
        for residue in topology.residues()
    ]
    selected: set[int] = set()
    for chain, resname, resnum, icode in residue_keys:
        for omm_key, index in omm_keys:
            omm_chain, omm_name, omm_num, omm_icode = omm_key
            if omm_name != resname or omm_num != resnum or omm_icode != icode:
                continue
            if omm_chain == chain or not omm_chain or not chain:
                selected.add(index)
                break
    return selected


def get_openmm_pdb_info(atom: Atom) -> str:
    """Get PDB info string for OpenMM atom."""
    chain, res_name, res_num, _ = pdb_residue_key_from_openmm(atom.residue)
    atom_name = atom.name.strip()
    return f"{chain}:{res_name}{res_num}:{atom_name}"


def build_rdkit_to_openmm_mapping(
    modeller_topology: Topology, rdkit_mol: Chem.Mol
) -> dict[int, int]:
    """
    Build a mapping from RDKit atom indices to OpenMM atom indices.

    Parameters
    ----------
    modeller_topology : openmm.app.Topology
        The topology from OpenMM Modeller
    rdkit_mol : rdkit.Chem.Mol
        The RDKit molecule from the same PDB

    Returns
    -------
    dict : mapping from RDKit atom index -> OpenMM atom index
    """
    # Build a dictionary of OpenMM atom identifiers to indices
    openmm_id_to_idx = {}
    for oa in modeller_topology.atoms():
        pdb_id = get_openmm_pdb_info(oa)
        openmm_id_to_idx[pdb_id] = oa.index

    # Build the mapping from RDKit index to OpenMM index
    rdkit_to_openmm = {}
    missing_atoms = []

    for ra in rdkit_mol.GetAtoms():
        rdkit_idx = ra.GetIdx()
        pdb_info = get_rdkit_pdb_info(ra)
        if pdb_info is None:
            missing_atoms.append(rdkit_idx)
            continue

        chain, res_name, res_num, atom_name = pdb_info
        pdb_id = f"{chain}:{res_name}{res_num}:{atom_name}"

        if pdb_id in openmm_id_to_idx:
            rdkit_to_openmm[rdkit_idx] = openmm_id_to_idx[pdb_id]
        else:
            missing_atoms.append(rdkit_idx)

    # Report any missing mappings
    if missing_atoms:
        logger.warning(f"Warning: Could not map {len(missing_atoms)} RDKit atoms to OpenMM")
        logger.warning(f"Missing RDKit atom indices: {missing_atoms}")

    return rdkit_to_openmm


def add_openmm_bonds_to_rdkit(modeller: Modeller, rdkit_mol: Chem.Mol) -> Chem.Mol:
    """Replace bonds in RDKit molecule with bonds from OpenMM topology."""
    rdkit2openmm = build_rdkit_to_openmm_mapping(modeller.topology, rdkit_mol)

    openmm2rdkit = {v: k for k, v in rdkit2openmm.items()}

    editable = Chem.RWMol(rdkit_mol)

    for bond in list(editable.GetBonds()):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        editable.RemoveBond(i, j)

    for bond in modeller.topology.bonds():
        # Get OpenMM atom info
        omm_atom1 = bond[0]
        omm_atom2 = bond[1]

        rdkit_idx1 = openmm2rdkit[omm_atom1.index]
        rdkit_idx2 = openmm2rdkit[omm_atom2.index]

        editable.AddBond(rdkit_idx1, rdkit_idx2, Chem.BondType.SINGLE)

    return editable.GetMol()
