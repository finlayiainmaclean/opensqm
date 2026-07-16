"""Regression tests for residues whose main variant name matches the PDB name."""

from openmm.app import PDBFile

from opensqm.cph.constantph import ConstantPH
from opensqm.cph.reference_energy.build_transitions import _resolve_named_transitions
from opensqm.cph.reference_energy.finder import ReferenceEnergyFinder, _make_pair_reference
from opensqm.cph.reference_energy.model_compounds import MODEL_COMPOUNDS
from opensqm.cph.reference_energy.models import TitratableResidueReference
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


def _his_reference() -> TitratableResidueReference:
    """A HIS reference with the real variant/transition topology (dummy energies)."""
    info = MODEL_COMPOUNDS["HIS"]
    variants = list(info["variants"])  # ["HIP", "HID", "HIE"]
    return TitratableResidueReference(
        residue_name="HIS",
        main_variant="HIP",
        variant_names=variants,
        variants=variants,
        charges=list(info["charges"]),  # [1, 0, 0]
        reference_energies_kj_per_mole=[0.0, -5.0, -4.0],
        transitions=_resolve_named_transitions(list(info["transitions"]), variants),
        ring_flip_bonds=[],
    )


def test_his_variant_mask_forbids_charged_hip_but_allows_tautomer_flip() -> None:
    """A masked neutral histidine builds all states, starts neutral, never proposes HIP.

    ``allowed_variant_indices`` restricts the residue to its two neutral
    tautomers (HID/HIE, indices 1/2). All three variants are still built, but
    the charged HIP (index 0, the ``protonated_index``) must never be the
    starting state nor a Monte-Carlo proposal target, while HID<->HIE stays
    available.
    """
    pdb = PDBFile("opensqm/cph/model-compounds/HIS.pdb")
    his_index = next(r.index for r in pdb.topology.residues() if r.name == "HIS")
    cph = ConstantPH(
        topology=pdb.topology,
        positions=pdb.positions,
        ph=7.0,
        config=ConstantpHSettings(),
        references={"HIS": _his_reference()},
        titratable_residue_indices=[his_index],
        allowed_variant_indices={his_index: [1, 2]},
        ring_flip_angles=None,
    )
    titration = cph.titrations[his_index]

    # All three protonation states are still parametrised.
    assert len(titration.explicit_states) == 3
    assert titration.allowed_state_indices == {1, 2}
    # The default start is protonated_index (HIP=0); it must be normalised into
    # the allowed neutral-tautomer set.
    assert titration.protonated_index == 0
    assert titration.current_index in (1, 2)

    # From either neutral tautomer the MC only ever proposes the other neutral
    # tautomer - never the masked-out charged HIP.
    for current in (1, 2):
        titration.current_index = current
        proposals = {ConstantPH._select_new_state(titration) for _ in range(200)}
        assert proposals == {3 - current}, proposals
        assert 0 not in proposals


def test_his_without_mask_can_reach_charged_hip() -> None:
    """Without a mask, an unrestricted histidine can propose all three variants."""
    pdb = PDBFile("opensqm/cph/model-compounds/HIS.pdb")
    his_index = next(r.index for r in pdb.topology.residues() if r.name == "HIS")
    cph = ConstantPH(
        topology=pdb.topology,
        positions=pdb.positions,
        ph=7.0,
        config=ConstantpHSettings(),
        references={"HIS": _his_reference()},
        titratable_residue_indices=[his_index],
        ring_flip_angles=None,
    )
    titration = cph.titrations[his_index]
    assert titration.allowed_state_indices is None
    titration.current_index = 1  # HID
    proposals = {ConstantPH._select_new_state(titration) for _ in range(300)}
    assert proposals == {0, 2}  # can reach both HIP and HIE
