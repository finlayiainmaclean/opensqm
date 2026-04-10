"""Monte Carlo moves that rigidly rotate terminal groups (e.g. ring flips)."""

import numpy as np
from loguru import logger
from openmm import app, unit


class TerminalRingMC:
    """Monte Carlo rigid rotation of terminal groups (e.g. ring flips) around a bond axis."""

    def __init__(
        self,
        simulation: app.Simulation,
        topology: app.Topology,
        k_bt: unit.Quantity,
        terminal_list: list[tuple[float, list[int]]] | None = None,
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
            Definitions of terminal groups to flip. Each entry is
            ``(angle_degrees, [axis_atom, pivot_atom, mobile_atom, ...])``:
            rotation axis is the bond from ``axis_atom`` to ``pivot_atom``; ``pivot_atom`` is the
            fixed point and all ``mobile_atom`` indices are rigidly rotated.
        """
        self.simulation = simulation
        self.topology = topology
        self.k_bt = k_bt
        self.terminal_list = terminal_list or []
        self.temperature = k_bt / unit.MOLAR_GAS_CONSTANT_R

    def rotate_terminal(self, terminal_res_index: int) -> None:
        """
        Rotate one terminal group by ``±angle_degrees`` (sign random).

        Uses Rodrigues' rotation about the axis from ``axis_atom`` to ``pivot_atom``,
        with the pivot fixed.

        Parameters
        ----------
        terminal_res_index :
            Index into ``self.terminal_list``.
        """
        angle, atom_indices = self.terminal_list[terminal_res_index]
        angle = float(np.random.choice([-1, 1])) * angle

        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        p0 = positions[atom_indices[0]]
        p1 = positions[atom_indices[1]]

        axis = p1 - p0
        axis = axis / np.linalg.norm(axis)

        theta = np.deg2rad(angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        for idx in atom_indices[2:]:
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
