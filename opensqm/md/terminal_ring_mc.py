"""Monte Carlo moves that rigidly rotate terminal groups (e.g. ring flips)."""

from dataclasses import dataclass

import numpy as np
from loguru import logger
from openmm import app, unit


@dataclass
class TerminalGroup:
    """Structure representing a terminal group for MC rotation."""

    angle: float
    bond: tuple[int, int]
    rotatable_group: list[int]


def find_terminal_group(
    topology: app.Topology, bond_atom_a: int, bond_atom_b: int, angle: float = 180.0
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

    return TerminalGroup(angle=float(angle), bond=(anchor, pivot), rotatable_group=mobile_atoms)


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
        angle_rad = float(np.random.choice([-1, 1])) * group.angle

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
        if delta_e <= 0.0 or np.random.random() < np.exp(-delta_e):
            logger.info(
                "TerminalRingMC: dihe_{idx}, {exp:.2f}, Accepted.",
                idx=terminal_res_index,
                exp=float(np.exp(-delta_e)),
            )
            self.simulation.context.setVelocitiesToTemperature(self.temperature)
        else:
            self.simulation.context.setPositions(old_positions)
            logger.info(
                "TerminalRingMC: dihe_{idx}, {exp:.2f}, Rejected.",
                idx=terminal_res_index,
                exp=float(np.exp(-delta_e)),
            )
