"""Tests for the terminal ring MC flipping module."""

from pathlib import Path

import numpy as np
import pytest
from openmm import app, openmm, unit

from opensqm.md.terminal_ring_mc import (
    TerminalRingMC,
    find_residue_ring_bond,
    find_terminal_group,
)

platform_ref = openmm.Platform.getPlatformByName("Reference")

# ACE-HIS-NME capped dipeptide, checked into the repo for constant-pH reference-energy
# generation (opensqm/cph/reference_energy) - reused here as a small, always-available
# vacuum system with a real HIS residue to rotate.
_MODEL_COMPOUND_PDB = (
    Path(__file__).resolve().parents[1] / "opensqm" / "cph" / "model-compounds" / "HIS.pdb"
)


class TestRotateTerminal:
    """Tests for the MC terminal rotation logic."""

    @pytest.fixture(autouse=True)
    def setup_system(self):
        """Build a vacuum ACE-HIS-NME system and its HIS ring-flip terminal group.

        Uses :func:`opensqm.md.terminal_ring_mc.find_residue_ring_bond` to resolve the
        CB-CG bond by residue name/number, exactly as production code
        (``opensqm.fix.enumerate_residue_flip_variants``) does, rather than hardcoding
        atom indices.
        """
        pdb = app.PDBFile(str(_MODEL_COMPOUND_PDB))
        # The checked-in file is this ACE-HIS-NME dipeptide solvated in a TIP3P box for
        # reference-energy generation; strip the water back out for a small vacuum system.
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.delete([res for res in modeller.topology.residues() if res.name == "HOH"])
        self.topology = modeller.topology
        forcefield = app.ForceField("amber/ff14SB.xml")
        system = forcefield.createSystem(
            self.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
        )

        integrator = openmm.LangevinMiddleIntegrator(
            298.15 * unit.kelvin, 1.0 * unit.picosecond**-1, 2.0 * unit.femtosecond
        )
        simulation = app.Simulation(self.topology, system, integrator, platform_ref)
        simulation.context.setPositions(modeller.positions)

        his_residue = next(res for res in self.topology.residues() if res.name == "HIS")
        anchor_idx, pivot_idx = find_residue_ring_bond(
            self.topology, "HIS", his_residue.id, his_residue.chain.id
        )
        derived_group = find_terminal_group(self.topology, anchor_idx, pivot_idx)
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

    def test_rotate_terminal(self):
        """Test that rotate_terminal executes correctly."""
        group = self.flipmc.terminal_list[0]
        pivot_idx = group.bond[1]  # CG - rotation centre
        axis_idx = group.bond[0]  # CB - axis start
        mobile = group.rotatable_group  # imidazole ring atoms
        assert mobile, "expected a non-empty rotatable group for the HIS CB-CG bond"

        pos_before = self._get_positions()

        self.flipmc.rotate_terminal(0)

        pos_after = self._get_positions()

        # Axis and pivot atoms must NOT move
        np.testing.assert_allclose(
            pos_after[axis_idx],
            pos_before[axis_idx],
            atol=1e-5,
            err_msg="Axis-start atom (CB) should not move",
        )
        np.testing.assert_allclose(
            pos_after[pivot_idx],
            pos_before[pivot_idx],
            atol=1e-5,
            err_msg="Pivot atom (CG) should not move",
        )

        # All mobile atoms must have moved
        for idx in mobile:
            assert not np.allclose(pos_after[idx], pos_before[idx], atol=1e-5), (
                f"Mobile atom {idx} should have moved after 180 degree rotation"
            )

        # A second 180 degree rotation must restore the mobile atoms
        self.flipmc.rotate_terminal(0)
        pos_restored = self._get_positions()
        np.testing.assert_allclose(
            pos_restored[mobile],
            pos_before[mobile],
            atol=1e-5,
            err_msg="Two 180 degree rotations should restore all mobile atom positions",
        )

        # Basic functional test that move_dihe works as well
        self.flipmc.move_dihe()
        self.flipmc.move_dihe()
