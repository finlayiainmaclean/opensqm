"""Regression tests for residues whose main variant name matches the PDB name."""

from openmm.app import PDBFile

from opensqm.cph.constantph import ConstantPH
from opensqm.cph.reference_energy.finder import ReferenceEnergyFinder, _make_pair_reference
from opensqm.cph.simulation_config import ConstantpHSettings


def test_cys_pairwise_reference_builds_two_states() -> None:
    """CYS/CYX must yield two explicit states for the pairwise reference finder."""
    pdb = PDBFile("opensqm/cph/model-compounds/CYS.pdb")
    residues = list(pdb.topology.residues())
    pair_reference = _make_pair_reference(
        residue_name="CYS",
        pair_variants=["CYS", "CYX"],
        pair_names=["CYS", "CYX"],
        pair_charges=[0, -1],
        pka=8.33,
    )
    cph = ConstantPH(
        topology=pdb.topology,
        positions=pdb.positions,
        ph=7.0,
        config=ConstantpHSettings(),
        references={residues[1].name: pair_reference},
        titratable_residue_indices=[1],
        ring_flip_angles=None,
    )
    titration = cph.titrations[1]
    assert len(titration.explicit_states) == 2
    assert [s.num_hydrogens for s in titration.explicit_states] == [5, 4]
    ReferenceEnergyFinder(cph, pka=8.33, temperature=300)
