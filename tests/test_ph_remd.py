"""Tests for pH replica-exchange acceptance and state bookkeeping."""

import math

import numpy as np
from openmm.app import PDBFile

from opensqm.cph.constantph import ConstantPH
from opensqm.cph.ph_remd import (
    ConstantPHRemd,
    num_titratable_protons,
    replica_exchange_log_probability,
)
from opensqm.cph.reference_energy.finder import _make_pair_reference
from opensqm.cph.simulation_config import ConstantpHSettings


def _build_cys_cph(ph: float, *, state_index: int = 0) -> ConstantPH:
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
        ph=ph,
        config=ConstantpHSettings(),
        references={residues[1].name: pair_reference},
        titratable_residue_indices=[1],
        ring_flip_angles=None,
    )
    cph.set_residue_state(1, state_index, relax=False)
    return cph


def test_replica_exchange_probability_same_proton_count() -> None:
    """Equal N gives unit acceptance regardless of pH separation."""
    cph_low = _build_cys_cph(4.0, state_index=0)
    cph_high = _build_cys_cph(10.0, state_index=0)
    assert num_titratable_protons(cph_low) == num_titratable_protons(cph_high)
    assert replica_exchange_log_probability(cph_low, cph_high) == 0.0


def test_replica_exchange_probability_sign() -> None:
    """A swap is favoured only when it moves the protonated config to lower pH."""
    # Already-matched arrangement (protonated at low pH, deprotonated at high pH):
    # exchanging them is unfavourable, so log-acceptance must be negative.
    protonated_low = _build_cys_cph(4.0, state_index=0)
    deprotonated_high = _build_cys_cph(10.0, state_index=1)
    assert num_titratable_protons(protonated_low) > num_titratable_protons(deprotonated_high)
    log_prob = replica_exchange_log_probability(protonated_low, deprotonated_high)
    assert log_prob < 0.0

    # Mismatched arrangement (deprotonated at low pH, protonated at high pH):
    # exchanging fixes it, so log-acceptance must be positive.
    deprotonated_low = _build_cys_cph(4.0, state_index=1)
    protonated_high = _build_cys_cph(10.0, state_index=0)
    log_prob = replica_exchange_log_probability(deprotonated_low, protonated_high)
    expected = (
        (num_titratable_protons(deprotonated_low) - num_titratable_protons(protonated_high))
        * math.log(10.0)
        * (4.0 - 10.0)
    )
    assert log_prob == expected
    assert log_prob > 0.0


def test_remd_single_replica_defaults_to_lowest_ph() -> None:
    pdb = PDBFile("opensqm/cph/model-compounds/CYS.pdb")
    residues = list(pdb.topology.residues())
    pair_reference = _make_pair_reference(
        residue_name="CYS",
        pair_variants=["CYS", "CYX"],
        pair_names=["CYS", "CYX"],
        pair_charges=[0, -1],
        pka=8.33,
    )
    ladder = [4.0, 7.0, 10.0]
    remd = ConstantPHRemd(
        topology=pdb.topology,
        positions=pdb.positions,
        ph=ladder,
        config=ConstantpHSettings(),
        references={residues[1].name: pair_reference},
        titratable_residue_indices=[1],
        n_replicas=1,
        ring_flip_angles=None,
    )
    assert remd.current_ph_values() == [4.0]
    assert remd.attempt_adjacent_exchanges() == []


def test_remd_initialises_two_replicas_on_ladder_endpoints() -> None:
    pdb = PDBFile("opensqm/cph/model-compounds/CYS.pdb")
    residues = list(pdb.topology.residues())
    pair_reference = _make_pair_reference(
        residue_name="CYS",
        pair_variants=["CYS", "CYX"],
        pair_names=["CYS", "CYX"],
        pair_charges=[0, -1],
        pka=8.33,
    )
    ladder = [4.0, 7.0, 10.0]
    remd = ConstantPHRemd(
        topology=pdb.topology,
        positions=pdb.positions,
        ph=ladder,
        config=ConstantpHSettings(),
        references={residues[1].name: pair_reference},
        titratable_residue_indices=[1],
        n_replicas=2,
        ring_flip_angles=None,
    )
    assert remd.current_ph_values() == [4.0, 10.0]
    assert len(remd.replicas) == 2


def test_remd_swap_exchanges_configuration_keeping_ph_fixed() -> None:
    """An accepted swap moves configurations between rungs; each replica keeps its pH."""
    ladder = [4.0, 10.0]

    pdb = PDBFile("opensqm/cph/model-compounds/CYS.pdb")
    residues = list(pdb.topology.residues())
    pair_reference = _make_pair_reference(
        residue_name="CYS",
        pair_variants=["CYS", "CYX"],
        pair_names=["CYS", "CYX"],
        pair_charges=[0, -1],
        pka=8.33,
    )
    remd = ConstantPHRemd(
        topology=pdb.topology,
        positions=pdb.positions,
        ph=ladder,
        config=ConstantpHSettings(),
        references={residues[1].name: pair_reference},
        titratable_residue_indices=[1],
        n_replicas=2,
        ring_flip_angles=None,
    )
    # Mismatched arrangement: deprotonated (state 1) at low pH, protonated
    # (state 0) at high pH. The swap is favourable and always accepted.
    remd.replicas[0].set_residue_state(1, 1, relax=False)
    remd.replicas[1].set_residue_state(1, 0, relax=False)
    remd.replicas[0].currentPHIndex = 0
    remd.replicas[1].currentPHIndex = 1

    np.random.seed(0)
    assert remd.attempt_replica_exchange(0, 1)

    # Configurations exchanged...
    assert remd.replicas[0].titrations[1].current_index == 0
    assert remd.replicas[1].titrations[1].current_index == 1
    # ...but each replica still samples its own pH rung.
    assert remd.replicas[0].currentPHIndex == 0
    assert remd.replicas[1].currentPHIndex == 1
    assert remd.current_ph_values() == [4.0, 10.0]


def test_simulated_tempering_disabled_pins_ph_index() -> None:
    """With simulated tempering off, MC steps never change the replica's pH index."""
    pdb = PDBFile("opensqm/cph/model-compounds/CYS.pdb")
    residues = list(pdb.topology.residues())
    pair_reference = _make_pair_reference(
        residue_name="CYS",
        pair_variants=["CYS", "CYX"],
        pair_names=["CYS", "CYX"],
        pair_charges=[0, -1],
        pka=8.33,
    )
    ladder = [4.0, 10.0]
    # Weights that would strongly pull a tempering move to index 1 if it ran, so
    # the pinned/tempered outcomes are unambiguous.
    biased_weights = [0.0, 1000.0]

    def _make(simulated_tempering: bool) -> ConstantPH:
        cph = ConstantPH(
            topology=pdb.topology,
            positions=pdb.positions,
            ph=ladder,
            config=ConstantpHSettings(),
            references={residues[1].name: pair_reference},
            titratable_residue_indices=[1],
            ring_flip_angles=None,
            weights=biased_weights,
            simulated_tempering=simulated_tempering,
        )
        cph.currentPHIndex = 0
        return cph

    pinned = _make(simulated_tempering=False)
    assert pinned.simulated_tempering is False
    np.random.seed(0)
    for _ in range(10):
        pinned.attempt_mc_step()
    assert pinned.currentPHIndex == 0

    # Sanity check the opposite: with tempering on the biased weights move it.
    tempered = _make(simulated_tempering=True)
    np.random.seed(0)
    tempered.attempt_mc_step()
    assert tempered.currentPHIndex == 1


def test_remd_forwards_simulated_tempering_to_replicas() -> None:
    """``simulated_tempering=False`` reaches every underlying replica."""
    pdb = PDBFile("opensqm/cph/model-compounds/CYS.pdb")
    residues = list(pdb.topology.residues())
    pair_reference = _make_pair_reference(
        residue_name="CYS",
        pair_variants=["CYS", "CYX"],
        pair_names=["CYS", "CYX"],
        pair_charges=[0, -1],
        pka=8.33,
    )
    ladder = [4.0, 7.0, 10.0]
    remd = ConstantPHRemd(
        topology=pdb.topology,
        positions=pdb.positions,
        ph=ladder,
        config=ConstantpHSettings(),
        references={residues[1].name: pair_reference},
        titratable_residue_indices=[1],
        n_replicas=len(ladder),
        ring_flip_angles=None,
        simulated_tempering=False,
    )
    assert [r.simulated_tempering for r in remd.replicas] == [False, False, False]
    # One fixed-pH replica per rung.
    assert remd.current_ph_values() == ladder


def test_remd_checkpoint_roundtrip(tmp_path) -> None:
    ladder = [4.0, 7.0, 10.0]
    pdb = PDBFile("opensqm/cph/model-compounds/CYS.pdb")
    residues = list(pdb.topology.residues())
    pair_reference = _make_pair_reference(
        residue_name="CYS",
        pair_variants=["CYS", "CYX"],
        pair_names=["CYS", "CYX"],
        pair_charges=[0, -1],
        pka=8.33,
    )
    references = {residues[1].name: pair_reference}
    titratable = [1]

    remd = ConstantPHRemd(
        topology=pdb.topology,
        positions=pdb.positions,
        ph=ladder,
        config=ConstantpHSettings(),
        references=references,
        titratable_residue_indices=titratable,
        n_replicas=2,
        ring_flip_angles=None,
    )
    remd.set_weights([0.0, 1.5, 3.0])
    remd.step(25)
    remd.replicas[0].set_residue_state(1, 1, relax=False)
    remd.replicas[0].currentPHIndex = 2
    remd.replicas[1].currentPHIndex = 0
    expected_steps = remd.replicas[0].simulation.currentStep

    ckpt_dir = tmp_path / "checkpoints"
    remd.save_checkpoint(ckpt_dir)

    restored = ConstantPHRemd(
        topology=pdb.topology,
        positions=pdb.positions,
        ph=ladder,
        config=ConstantpHSettings(),
        references=references,
        titratable_residue_indices=titratable,
        n_replicas=2,
        ring_flip_angles=None,
    )
    restored.load_checkpoint(ckpt_dir)

    assert restored.weights == remd.weights
    assert restored.replicas[0].simulation.currentStep == expected_steps
    assert restored.replicas[0].titrations[1].current_index == 1
    assert restored.replicas[0].currentPHIndex == 2
    assert restored.replicas[1].currentPHIndex == 0
