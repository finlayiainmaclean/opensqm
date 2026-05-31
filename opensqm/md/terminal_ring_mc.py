"""Monte Carlo moves that rigidly rotate terminal groups (e.g. ring flips)."""

from dataclasses import dataclass

import numpy as np
from openmm import app, unit

# Residue name -> (anchor_atom_name, pivot_atom_name) for the side-chain bond
# whose 180-degree rotation usefully reorients the planar terminal group. The
# pivot is the atom on the ring/amide side; the anchor stays on the backbone
# side. Aromatic residues (HIS/TYR/TRP) flip about Cbeta-Cgamma; the terminal
# amides of ASN/GLN flip about the bond directly preceding the amide carbon
# (CB-CG for ASN, CG-CD for GLN) which swaps the carbonyl O and the amide N
# without changing connectivity - the classic "MolProbity flip" that fixes
# the O/N ambiguity inherited from X-ray refinement.
RING_FLIP_BONDS: dict[str, tuple[str, str]] = {
    "HIS": ("CB", "CG"),
    "ASN": ("CB", "CG"),
    "GLN": ("CG", "CD"),
}


@dataclass
class TerminalGroup:
    """Structure representing a terminal group for MC rotation."""

    angles: list[float]
    bond: tuple[int, int]
    rotatable_group: list[int]


def find_residue_ring_bond(
    topology: app.Topology,
    residue_name: str,
    residue_number: int | str,
    chain_id: str | None = None,
) -> tuple[int, int]:
    """
    Resolve the bond atom indices whose rotation flips a residue's planar side-chain group.

    Supports the residues defined in :data:`RING_FLIP_BONDS`: aromatic side
    chains (HIS/TYR/TRP) flip about the CB-CG bond, and the terminal amides of
    ASN (CB-CG) and GLN (CG-CD) flip to swap the carbonyl O with the amide N
    -- the standard "MolProbity flip". The returned tuple is in
    ``(anchor, pivot)`` order suitable for :func:`find_terminal_group`.

    Parameters
    ----------
    topology :
        The OpenMM topology containing the residue.
    residue_name :
        Three-letter residue code (e.g. ``"HIS"``).
    residue_number :
        Residue sequence number (PDB ``resSeq``) of the target residue.
    chain_id :
        Optional chain identifier to disambiguate residues sharing the same
        ``resSeq`` across chains.

    Returns
    -------
    tuple[int, int]
        ``(anchor_atom_index, pivot_atom_index)`` for the rotatable bond.
    """
    key = residue_name.upper()
    if key not in RING_FLIP_BONDS:
        raise ValueError(
            f"No ring-flip bond defined for residue {residue_name!r}; "
            f"supported residues: {sorted(RING_FLIP_BONDS)}"
        )
    anchor_name, pivot_name = RING_FLIP_BONDS[key]

    target_id = str(residue_number)
    matches = [
        res
        for res in topology.residues()
        if res.name.upper() == key
        and str(res.id) == target_id
        and (chain_id is None or res.chain.id == chain_id)
    ]
    if not matches:
        raise ValueError(
            f"Residue {residue_name}.{residue_number} not found in topology"
            + (f" (chain {chain_id})" if chain_id else "")
        )
    if len(matches) > 1:
        chains = sorted({res.chain.id for res in matches})
        raise ValueError(
            f"Residue {residue_name}.{residue_number} is ambiguous across "
            f"chains {chains}; pass chain_id to disambiguate"
        )

    atoms_by_name = {atom.name: atom.index for atom in matches[0].atoms()}
    missing = [n for n in (anchor_name, pivot_name) if n not in atoms_by_name]
    if missing:
        raise ValueError(
            f"Residue {residue_name}.{residue_number} is missing atom(s) {missing}; "
            f"found atoms: {sorted(atoms_by_name)}"
        )
    return atoms_by_name[anchor_name], atoms_by_name[pivot_name]


def find_terminal_group(
    topology: app.Topology, bond_atom_a: int, bond_atom_b: int, angles: list[float] | None = None
) -> TerminalGroup:
    """
    Given an OpenMM Topology and a bond defined by two atom indices.

    Splits the molecule at the bond and robustly returns the atom indices
    formatted as a TerminalGroup.
    """
    # Build adjacency list from topology bonds
    adj = {atom.index: set() for atom in topology.atoms()}
    for bond in topology.bonds():
        adj[bond[0].index].add(bond[1].index)
        adj[bond[1].index].add(bond[0].index)

    if bond_atom_b not in adj[bond_atom_a]:
        raise ValueError(f"No bond found between atoms {bond_atom_a} and {bond_atom_b}")

    # Temporarily slice the bond to partition the graph
    adj[bond_atom_a].remove(bond_atom_b)
    adj[bond_atom_b].remove(bond_atom_a)

    def get_component(start_node):
        visited = {start_node}
        queue = [start_node]
        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    comp_a = get_component(bond_atom_a)
    comp_b = get_component(bond_atom_b)

    # Identify which side is smaller
    if len(comp_a) < len(comp_b):
        smaller_side = comp_a
        pivot = bond_atom_a
        anchor = bond_atom_b
    else:
        smaller_side = comp_b
        pivot = bond_atom_b
        anchor = bond_atom_a

    # Process the mobile atoms exclusively
    mobile_atoms = sorted(smaller_side)
    mobile_atoms.remove(pivot)

    if angles is None:
        angles = [180.0]
    return TerminalGroup(angles=angles, bond=(anchor, pivot), rotatable_group=mobile_atoms)


class TerminalRingMC:
    """Monte Carlo rigid rotation of terminal groups (e.g. ring flips) around a bond axis."""

    def __init__(
        self,
        simulation: app.Simulation,
        topology: app.Topology,
        k_bt: unit.Quantity,
        terminal_list: list[TerminalGroup] | None = None,
    ) -> None:
        """
        Configure the MC mover for the given simulation and thermal energy.

        Parameters
        ----------
        simulation :
            The OpenMM Simulation object.
        topology :
            The OpenMM Topology object.
        k_bt :
            Thermal energy in energy units, e.g. ``unit.MOLAR_GAS_CONSTANT_R * 300 * unit.kelvin``.
        terminal_list :
            A list of TerminalGroup dataclasses detailing the rotation axis and mobile atoms.
        """
        self.simulation = simulation
        self.topology = topology
        self.k_bt = k_bt
        self.terminal_list = terminal_list or []
        self.temperature = k_bt / unit.MOLAR_GAS_CONSTANT_R
        self.n_attempts = 0
        self.n_accepted = 0

    @property
    def acceptance_rate(self) -> float:
        """Return the fraction of accepted moves (0.0 if no attempts yet)."""
        return self.n_accepted / self.n_attempts if self.n_attempts > 0 else 0.0

    def reset_stats(self) -> None:
        """Reset the acceptance counters."""
        self.n_attempts = 0
        self.n_accepted = 0

    def rotate_terminal(self, terminal_res_index: int) -> None:
        """
        Rotate one terminal group by ``±angle_degrees`` (sign random).

        Uses Rodrigues' rotation about the axis from ``anchor`` to ``pivot``,
        with the pivot fixed.

        Parameters
        ----------
        terminal_res_index :
            Index into ``self.terminal_list``.
        """
        group = self.terminal_list[terminal_res_index]
        angle_deg = float(np.random.choice(group.angles))
        angle_rad = float(np.random.choice([-1, 1])) * angle_deg

        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        p0 = positions[group.bond[0]]
        p1 = positions[group.bond[1]]

        axis = p1 - p0
        axis = axis / np.linalg.norm(axis)

        theta = np.deg2rad(angle_rad)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        for idx in group.rotatable_group:
            v = positions[idx] - p1
            v_rot = v * cos_t + np.cross(axis, v) * sin_t + axis * np.dot(axis, v) * (1.0 - cos_t)
            positions[idx] = p1 + v_rot

        self.simulation.context.setPositions(positions * unit.nanometer)

    def move_dihe(self) -> None:
        """
        One Metropolis MC move: pick a terminal group at random, propose ±rotation, accept/reject.

        On accept, velocities are redrawn at ``self.temperature``; on reject, positions revert.
        """
        terminal_res_index = int(np.random.randint(len(self.terminal_list)))

        state_old = self.simulation.context.getState(getPositions=True, getEnergy=True)
        old_positions = state_old.getPositions(asNumpy=True)
        e_old = state_old.getPotentialEnergy() / self.k_bt

        self.rotate_terminal(terminal_res_index)

        e_new = self.simulation.context.getState(getEnergy=True).getPotentialEnergy() / self.k_bt

        delta_e = e_new - e_old
        self.n_attempts += 1
        if delta_e <= 0.0 or np.random.random() < np.exp(-delta_e):
            self.n_accepted += 1
            self.simulation.context.setVelocitiesToTemperature(self.temperature)
        else:
            self.simulation.context.setPositions(old_positions)
