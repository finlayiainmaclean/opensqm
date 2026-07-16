"""Tests for per-state DCD append on resume."""

from openmm import unit
from openmm.app import Modeller, PDBFile
from openmm.app.dcdfile import DCDFile


def test_dcd_append_requires_read_write_mode(tmp_path) -> None:
    pdb = PDBFile("opensqm/cph/model-compounds/CYS.pdb")
    modeller = Modeller(pdb.topology, pdb.positions)
    topology = modeller.topology
    positions = modeller.positions

    dcd_path = tmp_path / "traj.0.dcd"
    with dcd_path.open("wb") as handle:
        dcd = DCDFile(
            handle,
            topology,
            0.004 * unit.picoseconds,
            0,
            250,
            append=False,
        )
        dcd.writeModel(positions)

    with dcd_path.open("r+b") as handle:
        dcd = DCDFile(
            handle,
            topology,
            0.004 * unit.picoseconds,
            250,
            250,
            append=True,
        )
        dcd.writeModel(positions)

    assert dcd_path.stat().st_size > 0
