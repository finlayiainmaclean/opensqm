"""Tests for the terminal ring MC flipping module."""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pytest
from openmm import app, openmm, unit

from opensqm.md.terminal_ring_mc import TerminalRingMC, find_terminal_group

nonbonded_amber = {
    "nonbondedMethod": app.PME,
    "nonbondedCutoff": 1.0 * unit.nanometer,
    "constraints": app.HBonds,
}

platform_ref = openmm.Platform.getPlatformByName("Reference")


def load_amber_sys(
    inpcrd_file: Union[str, Path], prmtop_file: Union[str, Path], nonbonded_settings: dict
) -> Tuple[app.AmberInpcrdFile, app.AmberPrmtopFile, openmm.System]:
    """Load Amber system from inpcrd and prmtop file."""
    inpcrd = app.AmberInpcrdFile(str(inpcrd_file))
    prmtop = app.AmberPrmtopFile(str(prmtop_file), periodicBoxVectors=inpcrd.boxVectors)
    sys = prmtop.createSystem(**nonbonded_settings)
    return inpcrd, prmtop, sys


class TestRotateTerminal:
    """Tests for the MC terminal rotation logic."""

    @pytest.fixture(autouse=True)
    def setup_system(self):
        """Set up OpenMM system for testing."""
        self.base = Path(__file__).resolve().parent
        self.output = self.base / "output"
        self.output.mkdir(exist_ok=True)

        inpcrd, prmtop, system = load_amber_sys(
            self.base / "data" / "07_tip3p.inpcrd",
            self.base / "data" / "07_tip3p.prmtop",
            nonbonded_amber,
        )
        self.topology = prmtop.topology

        integrator = openmm.LangevinMiddleIntegrator(
            298.15 * unit.kelvin, 1.0 * unit.picosecond**-1, 2.0 * unit.femtosecond
        )
        simulation = app.Simulation(self.topology, system, integrator, platform_ref)
        simulation.context.setPositions(inpcrd.positions)

        # Uses programmatically derived TerminalGroup for the C15-C16 splitting
        derived_group = find_terminal_group(self.topology, 19, 20)
        assert derived_group.bond == (19, 20)
        assert derived_group.rotatable_group == [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

        terminal_list = [derived_group]

        kBT = 298.15 * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R

        self.flipmc = TerminalRingMC(
            simulation=simulation,
            topology=self.topology,
            k_bt=kBT,
            terminal_list=terminal_list,
        )

    def _get_positions(self):
        state = self.flipmc.simulation.context.getState(getPositions=True)
        return state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    def _save_pdb(self, filename):
        state = self.flipmc.simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
        positions = state.getPositions()
        path = self.output / filename
        with path.open("w") as f:
            app.PDBFile.writeFile(self.topology, positions, f)
        print(f"  Saved: {path}")

    def test_rotate_terminal(self):
        """Test that rotate_terminal executes correctly."""
        print("\n# Test rotate_terminal: 180° rotation around C15-C16 bond")
        group = self.flipmc.terminal_list[0]
        pivot_idx = group.bond[1]  # C16 - rotation centre
        axis_idx = group.bond[0]  # C15 - axis start
        mobile = group.rotatable_group  # phenyl ring atoms

        pos_before = self._get_positions()
        self._save_pdb("rotate_terminal_before.pdb")

        self.flipmc.rotate_terminal(0)

        pos_after = self._get_positions()
        self._save_pdb("rotate_terminal_after.pdb")

        # Axis and pivot atoms must NOT move
        np.testing.assert_allclose(
            pos_after[axis_idx],
            pos_before[axis_idx],
            atol=1e-5,
            err_msg="Axis-start atom (C15) should not move",
        )
        np.testing.assert_allclose(
            pos_after[pivot_idx],
            pos_before[pivot_idx],
            atol=1e-5,
            err_msg="Pivot atom (C16) should not move",
        )

        # All mobile atoms must have moved
        for idx in mobile:
            assert not np.allclose(pos_after[idx], pos_before[idx], atol=1e-5), (
                f"Mobile atom {idx} should have moved after 180° rotation"
            )

        # A second 180° must restore the mobile atoms
        self.flipmc.rotate_terminal(0)
        pos_restored = self._get_positions()
        np.testing.assert_allclose(
            pos_restored[mobile],
            pos_before[mobile],
            atol=1e-5,
            err_msg="Two 180° rotations should restore all mobile atom positions",
        )
        print("  PASSED: axis/pivot unmoved, mobile atoms rotated, double-180° restores positions")

        # Basic functional test that move_dihe works as well
        self.flipmc.move_dihe()
        self.flipmc.move_dihe()
