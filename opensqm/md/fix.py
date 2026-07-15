"""Prepare protein structures with PDBFixer: renumber chains, crop, and cap termini."""

# pyrefly: ignore [missing-import]
from pathlib import Path

import numpy as np

# pyrefly: ignore [missing-import]
from openmm import unit

# pyrefly: ignore [missing-import]
from openmm.app import Element, Modeller, PDBFile, Topology

# pyrefly: ignore [missing-import]
from openmm.app.topology import Atom, Chain, Residue

# pyrefly: ignore [missing-import]
from pdbfixer import PDBFixer

# pyrefly: ignore [missing-import]
from opensqm.cph.minimize import minimize

ALL_PROTEIN_RESNAMES = {
    # Standard 20 amino acids
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    # Protonation state variants
    "CYX",
    "HIP",
    "HID",
    "HIE",
    "HIN",
    "ASH",
    "GLH",
    "LYN",
    # Non-standard/modified amino acids
    "SEC",
    "PYL",
    "MSE",
    "HYP",
    "MLY",
    "PTR",
    "SEP",
    "TPO",
}


def renumber_chains(fixer: PDBFixer) -> PDBFixer:
    """Split chains whenever ``residue.id`` does not increase by 1.

    Return the fixer with a rebuilt topology whose chains are broken at
    every non-consecutive residue numbering step.
    """
    modeller = Modeller(topology=fixer.topology, positions=fixer.positions)
    old_top = modeller.topology
    new_top = Topology()

    # Map atoms to new topology atoms
    atom_map = {}

    for chain in old_top.chains():
        prev_resid = None
        new_chain = new_top.addChain()

        for res in chain.residues():
            resid = int(res.id)

            if prev_resid is not None and resid != prev_resid + 1:
                # break -> start new chain
                new_chain = new_top.addChain()

            new_res = new_top.addResidue(res.name, new_chain, res.id, res.insertionCode)

            for atom in res.atoms():
                new_atom = new_top.addAtom(atom.name, atom.element, new_res)
                atom_map[atom] = new_atom

            prev_resid = resid

    # Rebuild bonds - only within the same chain
    for bond in old_top.bonds():
        a1, a2 = bond

        new_a1, new_a2 = atom_map[a1], atom_map[a2]

        # Check if both atoms are in the same chain
        if new_a1.residue.chain == new_a2.residue.chain:
            new_top.addBond(new_a1, new_a2)

    fixer.topology = new_top
    fixer.positions = modeller.positions

    return fixer


def split_chains_at_breaks(fixer: PDBFixer, threshold: float = 4.0) -> PDBFixer:
    """
    Identify chain breaks in a Modeller object and split chains by updating chain indices.

    Parameters
    ----------
        modeller (Modeller): An OpenMM Modeller object containing the protein structure.
        threshold (float): Maximum distance (in Å) between consecutive C-alpha
            atoms to be considered continuous.

    Returns
    -------
        modeller: The updated Modeller object with new chain indices for split chains.
    """
    modeller = Modeller(fixer.topology, fixer.positions)

    topology = modeller.topology
    positions = modeller.positions

    new_topology = Topology()
    new_topology.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())

    chain_id_counter = 0  # Start with the first chain index

    for chain in topology.chains():
        # Get all residues in the chain
        residues = list(chain.residues())
        new_chain = None

        prev_residue = None
        prev_position = None

        for residue in residues:
            # Get the alpha carbon (CA) atom for distance calculation
            ca_atom = next((atom for atom in residue.atoms() if atom.name == "CA"), None)
            if ca_atom is None:
                continue  # Skip residues without a CA atom

            ca_position = positions[ca_atom.index]

            # Check if there's a break
            if prev_residue is not None:
                distance = np.linalg.norm(ca_position - prev_position)
                distance_value = distance.in_units_of(unit.angstrom)._value  # Convert to angstroms
                if distance_value > threshold:  # Threshold for detecting breaks
                    # Chain break detected, start a new chain
                    chain_id_counter += 1
                    new_chain = None

            # Add to the new topology
            if new_chain is None:
                new_chain = new_topology.addChain(f"chain{chain_id_counter}")

            # Add the residue to the current chain
            new_residue = new_topology.addResidue(residue.name, new_chain, residue.id)
            for atom in residue.atoms():
                # Add atom without manually setting its index
                new_topology.addAtom(atom.name, atom.element, new_residue)

            prev_residue = residue
            prev_position = ca_position

        # Increment chain ID for the next chain
        chain_id_counter += 1

    # # Create a new Modeller object with the updated topology
    # new_modeller = app.Modeller(new_topology, positions)

    fixer.topology = new_topology
    fixer.positions = positions
    return fixer


def is_protein_chain(chain: Chain) -> bool:
    """Return whether any residue in the chain is a known amino acid."""
    return any(res.name in ALL_PROTEIN_RESNAMES for res in chain.residues())


def calc_coordinate(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond_len: float,
    theta: float,
    di_angle: float,
) -> np.ndarray:
    """Calculate the position of a fourth atom ``d`` from atoms ``a``, ``b``, ``c``.

    bond_len: c-d bond length (angstrom)
    theta: b,c,d angle (degrees)
    di_angle: a,b,c,d dihedral (degrees).

    with thanks to https://github.com/osita-sunday-nnyigide/Pras_Server/
    """
    di_angle = np.deg2rad(di_angle)

    u = c - b
    x = a - b
    v = (x) - (np.dot((x), u) / np.dot(u, u)) * u

    w = np.cross(u, x)

    q = (v / np.linalg.norm(v)) * np.cos(di_angle)
    e = (w / np.linalg.norm(w)) * np.sin(di_angle)

    pos_temp2 = np.array((b + (q + e)))

    u1 = b - c
    y1 = pos_temp2 - c

    mag_y1 = np.linalg.norm(y1)
    mag_u1 = np.linalg.norm(u1)

    theta_bcd = np.arccos(np.dot(u1, y1) / (mag_u1 * mag_y1))
    rotate = np.deg2rad(theta) - theta_bcd

    z = np.cross(u1, y1)
    n = z / np.linalg.norm(z)

    pos_ini = (
        c
        + y1 * np.cos(rotate)
        + (np.cross(n, y1) * np.sin(rotate))
        + n * (np.dot(n, y1)) * (1 - np.cos(rotate))
    )

    return ((pos_ini - c) * (bond_len / np.linalg.norm(pos_ini - c))) + c


def get_atom_position(
    residue: Residue, atom_name: str, positions: unit.Quantity
) -> np.ndarray | None:
    """Get position of atom by name in a residue."""
    for atom in residue.atoms():
        if atom.name == atom_name:
            idx = atom.index
            return positions[idx].value_in_unit(unit.angstrom)
    return None


def get_nme_pos(end_residue: Residue, positions: unit.Quantity) -> tuple[np.ndarray, np.ndarray]:
    """Calculate NME cap positions."""
    pos_o = get_atom_position(end_residue, "O", positions)
    pos_ca = get_atom_position(end_residue, "CA", positions)
    pos_c = get_atom_position(end_residue, "C", positions)

    if pos_o is None or pos_ca is None or pos_c is None:
        raise ValueError(
            f"Could not find O, CA, or C atoms in residue {end_residue.id} for NME capping"
        )

    # Bisect the O, C, CA angle
    v1 = pos_o - pos_c
    v1 /= np.linalg.norm(v1)
    v2 = pos_ca - pos_c
    v2 /= np.linalg.norm(v2)
    bisector = v1 + v2
    bisector /= np.linalg.norm(bisector)

    # Apply translation to NME N atom from main chain C atom
    bondLength = 1.34
    N_position = bondLength * -(bisector) + pos_c

    # Place NME C position using bond length, angle, and dihedral
    bondLength = 1.45
    C_position = calc_coordinate(pos_o, pos_c, N_position, bondLength, 120, 0)

    return N_position, C_position


def get_ace_pos(
    start_residue: Residue, positions: unit.Quantity
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate ACE cap positions."""
    pos_c = get_atom_position(start_residue, "C", positions)
    pos_ca = get_atom_position(start_residue, "CA", positions)
    pos_n = get_atom_position(start_residue, "N", positions)

    if pos_c is None or pos_ca is None or pos_n is None:
        raise ValueError(
            f"Could not find C, CA, or N atoms in residue {start_residue.id} for ACE capping"
        )

    # C-CA-N-C dihedral has two minima at -60,60
    C_position = calc_coordinate(pos_c, pos_ca, pos_n, 1.34, 120, -60)
    CH3_position = calc_coordinate(pos_ca, pos_n, C_position, 1.52, 120, 180)
    O_position = calc_coordinate(pos_ca, pos_n, C_position, 1.23, 120, 0)

    return C_position, CH3_position, O_position


def add_ace_cap(chain: Chain, positions: unit.Quantity) -> tuple[Topology, list[unit.Quantity]]:
    """Add ACE cap to N-terminus of chain."""
    # Get first residue of chain
    first_residue = next(iter(chain.residues()))

    # Calculate ACE positions
    ace_c_pos, ace_ch3_pos, ace_o_pos = get_ace_pos(first_residue, positions)

    # Create new topology with ACE residue
    new_topology = Topology()
    new_chain = new_topology.addChain(chain.id)
    ace_residue = new_topology.addResidue("ACE", new_chain)

    # Add ACE atoms
    new_topology.addAtom("C", Element.getBySymbol("C"), ace_residue)
    new_topology.addAtom("CH3", Element.getBySymbol("C"), ace_residue)
    new_topology.addAtom("O", Element.getBySymbol("O"), ace_residue)

    # Create positions array
    ace_positions = [
        ace_c_pos * unit.angstrom,
        ace_ch3_pos * unit.angstrom,
        ace_o_pos * unit.angstrom,
    ]

    return new_topology, ace_positions


def add_nme_cap(chain: Chain, positions: unit.Quantity) -> tuple[Topology, list[unit.Quantity]]:
    """Add NME cap to C-terminus of chain."""
    # Get last residue of chain
    last_residue = list(chain.residues())[-1]

    # Calculate NME positions
    nme_n_pos, nme_c_pos = get_nme_pos(last_residue, positions)

    # Create new topology with NME residue
    new_topology = Topology()
    new_chain = new_topology.addChain(chain.id)
    nme_residue = new_topology.addResidue("NME", new_chain)

    # Add NME atoms
    new_topology.addAtom("N", Element.getBySymbol("N"), nme_residue)
    new_topology.addAtom("C", Element.getBySymbol("C"), nme_residue)

    # Create positions array
    nme_positions = [nme_n_pos * unit.angstrom, nme_c_pos * unit.angstrom]

    return new_topology, nme_positions


def n_terminus_needs_cap(chain: Chain) -> bool:
    """Check whether the N-terminus needs an ACE cap.

    Return ``True`` for an artificial cut (needs a cap), ``False`` for a
    natural NH3+ terminus.
    """
    # Get first residue
    first_residue = next(iter(chain.residues()))

    # Find the N-terminus nitrogen
    n_atom = None
    for atom in first_residue.atoms():
        if atom.name == "N":
            n_atom = atom
            break

    if n_atom is None:
        return False

    # Count hydrogens bonded to nitrogen
    h_count = 0
    for bond in chain.topology.bonds():
        if n_atom in bond:
            other_atom = bond[0] if bond[1] == n_atom else bond[1]
            if other_atom.element.symbol == "H":
                h_count += 1

    # Natural N-terminus: 3 hydrogens (NH3+) - NO cap needed
    # Truncated N-terminus: 1 or 0 hydrogens (NH or N) - NEEDS cap
    return h_count <= 1


def c_terminus_needs_cap(chain: Chain) -> bool:
    """Check whether the C-terminus needs an NME cap.

    Return ``True`` for an artificial cut (needs a cap), ``False`` for a
    natural COO- terminus.
    """
    # Get last residue
    residues = list(chain.residues())
    last_residue = residues[-1]

    # Find the C-terminus carbon
    c_atom = None
    for atom in last_residue.atoms():
        if atom.name == "C":
            c_atom = atom
            break

    if c_atom is None:
        return False

    # Count oxygens bonded to this carbon
    o_atoms = []
    for bond in chain.topology.bonds():
        if c_atom in bond:
            other_atom = bond[0] if bond[1] == c_atom else bond[1]
            if other_atom.element.symbol == "O":
                o_atoms.append(other_atom)

    # Check how many oxygens are double-bonded or singly bonded
    # (If OpenMM topology doesn't expose bond order, we infer from counts)
    # Natural C-terminus (COO-): 2 oxygens, no attached carbon next in chain → NO cap needed
    # Truncated C-terminus (C=O only, missing carbonyl O): 1 oxygen → NEEDS cap
    return len(o_atoms) < 2


def _should_skip_protein_atom(
    atom: Atom,
    *,
    is_first_residue: bool,
    need_ace: bool,
    need_nme: bool,
) -> bool:
    """Drop atoms incompatible with ACE/NME caps."""
    if atom.name == "OXT" and need_nme:
        return True
    if need_ace and is_first_residue and atom.name in {"H2", "H3", "HN2", "HN3", "HT2", "HT3"}:
        return True
    return False


class PDBFixer2(PDBFixer):
    """PDBFixer subclass that adds ACE/NME caps to protein chain termini."""

    def add_caps(
        self,
        minimise_caps: bool = False,
        ff_files: tuple = (
            "amber/ff14SB.xml",
            "amber/phosaa10.xml",
            "amber/tip3p_standard.xml",
            "implicit/gbn2.xml",
        ),
        force_caps: bool = False,
    ) -> None:
        """Add ACE/NME caps to protein chain termini that need them."""
        modeller = Modeller(self.topology, self.positions)
        final_topology = Topology()
        final_positions = []
        atom_offset = 0

        for chain in modeller.topology.chains():
            need_ace = force_caps or n_terminus_needs_cap(chain)
            need_nme = force_caps or c_terminus_needs_cap(chain)
            if is_protein_chain(chain) and (need_ace or need_nme):
                new_chain = final_topology.addChain(chain.id)
                atom_map = {}
                current_idx = atom_offset
                ace_atoms = {}
                ace_atom_indices_in_final_topology = {}
                nme_atoms = {}
                nme_atom_indices_in_final_topology = {}

                if need_ace:
                    ace_topology, ace_positions = add_ace_cap(chain, modeller.positions)
                    for residue in ace_topology.residues():
                        new_residue = final_topology.addResidue(residue.name, new_chain)
                        for atom in residue.atoms():
                            final_topology.addAtom(atom.name, atom.element, new_residue)
                            ace_atoms[atom.name] = current_idx
                            ace_atom_indices_in_final_topology[atom.name] = current_idx
                            current_idx += 1
                    final_positions.extend(ace_positions)

                    final_atoms = list(final_topology.atoms())
                    if (
                        "C" in ace_atom_indices_in_final_topology
                        and "CH3" in ace_atom_indices_in_final_topology
                    ):
                        final_topology.addBond(
                            final_atoms[ace_atom_indices_in_final_topology["C"]],
                            final_atoms[ace_atom_indices_in_final_topology["CH3"]],
                        )
                    if (
                        "C" in ace_atom_indices_in_final_topology
                        and "O" in ace_atom_indices_in_final_topology
                    ):
                        final_topology.addBond(
                            final_atoms[ace_atom_indices_in_final_topology["C"]],
                            final_atoms[ace_atom_indices_in_final_topology["O"]],
                        )

                first_residue_n_idx = None
                last_residue_c_idx = None
                chain_residues = list(chain.residues())
                for res_idx, residue in enumerate(chain_residues):
                    is_first_residue = res_idx == 0
                    new_residue = final_topology.addResidue(residue.name, new_chain)
                    for atom in residue.atoms():
                        if _should_skip_protein_atom(
                            atom,
                            is_first_residue=is_first_residue,
                            need_ace=need_ace,
                            need_nme=need_nme,
                        ):
                            continue
                        final_topology.addAtom(atom.name, atom.element, new_residue)
                        atom_map[atom.index] = current_idx
                        if first_residue_n_idx is None and atom.name == "N":
                            first_residue_n_idx = current_idx
                        if atom.name == "C":
                            last_residue_c_idx = current_idx
                        final_positions.append(modeller.positions[atom.index])
                        current_idx += 1

                if need_nme:
                    nme_topology, nme_positions = add_nme_cap(chain, modeller.positions)
                    for residue in nme_topology.residues():
                        new_residue = final_topology.addResidue(residue.name, new_chain)
                        for atom in residue.atoms():
                            final_topology.addAtom(atom.name, atom.element, new_residue)
                            nme_atoms[atom.name] = current_idx
                            nme_atom_indices_in_final_topology[atom.name] = current_idx
                            current_idx += 1
                    final_positions.extend(nme_positions)

                    final_atoms = list(final_topology.atoms())
                    if (
                        "N" in nme_atom_indices_in_final_topology
                        and "C" in nme_atom_indices_in_final_topology
                    ):
                        final_topology.addBond(
                            final_atoms[nme_atom_indices_in_final_topology["N"]],
                            final_atoms[nme_atom_indices_in_final_topology["C"]],
                        )

                final_atoms = list(final_topology.atoms())
                for bond in modeller.topology.bonds():
                    atom1, atom2 = bond
                    if (
                        atom1.residue.chain == chain
                        and atom2.residue.chain == chain
                        and atom1.index in atom_map
                        and atom2.index in atom_map
                    ):
                        if _should_skip_protein_atom(
                            atom1,
                            is_first_residue=atom1.residue == chain_residues[0],
                            need_ace=need_ace,
                            need_nme=need_nme,
                        ) or _should_skip_protein_atom(
                            atom2,
                            is_first_residue=atom2.residue == chain_residues[0],
                            need_ace=need_ace,
                            need_nme=need_nme,
                        ):
                            continue
                        final_topology.addBond(
                            final_atoms[atom_map[atom1.index]],
                            final_atoms[atom_map[atom2.index]],
                        )

                if need_ace and "C" in ace_atoms and first_residue_n_idx is not None:
                    final_topology.addBond(
                        final_atoms[ace_atoms["C"]],
                        final_atoms[first_residue_n_idx],
                    )
                if need_nme and last_residue_c_idx is not None and "N" in nme_atoms:
                    final_topology.addBond(
                        final_atoms[last_residue_c_idx],
                        final_atoms[nme_atoms["N"]],
                    )

                atom_offset = current_idx

            else:
                new_chain = final_topology.addChain(chain.id)
                atom_map = {}

                # Copy residues and atoms
                for residue in chain.residues():
                    new_residue = final_topology.addResidue(residue.name, new_chain)
                    for atom in residue.atoms():
                        final_topology.addAtom(atom.name, atom.element, new_residue)
                        atom_map[atom.index] = atom_offset
                        final_positions.append(modeller.positions[atom.index])
                        atom_offset += 1

                # Copy bonds *within* this chain
                final_atoms = list(final_topology.atoms())
                for bond in modeller.topology.bonds():
                    atom1, atom2 = bond
                    if atom1.residue.chain == chain and atom2.residue.chain == chain:
                        if atom1.index in atom_map and atom2.index in atom_map:
                            final_topology.addBond(
                                final_atoms[atom_map[atom1.index]],
                                final_atoms[atom_map[atom2.index]],
                            )

        # Create final modeller with capped structure
        # Convert list of Quantity objects to numpy array of values
        final_positions = np.array([pos.value_in_unit(unit.angstrom) for pos in final_positions])
        final_positions_quantity = unit.Quantity(final_positions, unit.angstrom)
        final_modeller = Modeller(final_topology, final_positions_quantity)
        if minimise_caps:
            final_modeller = minimize(final_modeller, ff_files=ff_files)

        self.positions = final_modeller.positions
        self.topology = final_modeller.topology


def _load_ligand_coords_angstrom(ligand_path: Path) -> np.ndarray:
    """Load ligand atom coordinates (Å) from a structure file."""
    from rdkit import Chem

    ligand_path = Path(ligand_path)
    suffix = ligand_path.suffix.lower()
    if suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(ligand_path), removeHs=False, sanitize=False)
    elif suffix in {".sdf", ".mol", ".mol2"}:
        mol = Chem.MolFromMolFile(str(ligand_path), removeHs=False)
    else:
        raise ValueError(
            f"Unsupported ligand format {suffix!r}; expected .pdb, .sdf, .mol, or .mol2"
        )
    if mol is None:
        raise ValueError(f"Could not read ligand coordinates from {ligand_path}")
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol, addCoords=True)
    conf = mol.GetConformer()
    return np.array(
        [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())],
        dtype=float,
    )


def crop_protein(
    fixer: PDBFixer,
    ligand_path: Path,
    distance_angstrom: float = 10.0,
) -> PDBFixer:
    """Keep residues with at least one atom within *distance_angstrom* of the ligand."""
    ligand_coords = _load_ligand_coords_angstrom(Path(ligand_path))
    pos_ang = np.array(
        [pos.value_in_unit(unit.angstrom) for pos in fixer.positions],
        dtype=float,
    )

    residues_to_keep = set()
    for residue in fixer.topology.residues():
        for atom in residue.atoms():
            distances = np.linalg.norm(ligand_coords - pos_ang[atom.index], axis=1)
            if np.any(distances < distance_angstrom):
                residues_to_keep.add(residue)
                break

    if not residues_to_keep:
        raise ValueError(f"No residues within {distance_angstrom} Å of ligand {ligand_path}")

    atoms_to_delete = [
        atom for atom in fixer.topology.atoms() if atom.residue not in residues_to_keep
    ]
    if atoms_to_delete:
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.delete(atoms_to_delete)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions
        print(f"Cropped to {len(residues_to_keep)} residues within {distance_angstrom} Å of ligand")

    return fixer


def run_pdbfixer(
    input_protein_path: Path,
    output_protein_path: Path,
    reference_ligand_path: Path | None = None,
    keep_waters: bool = True,
    keep_ions: bool = True,
    ph: float = 7.4,
) -> PDBFixer:
    """Fix a protein PDB and write the prepared structure to disk.

    Optionally crop around a reference ligand, preserve waters/ions,
    add missing residues/atoms, renumber chains, cap termini, and protonate
    at the given pH. Return the resulting fixer.
    """
    input_protein_path = Path(input_protein_path)
    output_protein_path = Path(output_protein_path)

    fixer = PDBFixer2(filename=str(input_protein_path))

    if reference_ligand_path is not None:
        fixer = crop_protein(fixer, reference_ligand_path, 12.0)

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
    fixer.addMissingAtoms()

    fixer1 = renumber_chains(fixer)
    fixer.topology, fixer.positions = fixer1.topology, fixer1.positions

    fixer.add_caps(force_caps=False)
    fixer.addMissingHydrogens(ph)

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

        print(f"Added {len(ion_atoms)} zinc atoms back to the structure")

    PDBFile.writeFile(fixer.topology, fixer.positions, output_protein_path.open("w"), keepIds=True)
    return fixer


if __name__ == "__main__":
    run_pdbfixer("t.pdb", "/tmp/prot.pdb", keep_waters=True)
