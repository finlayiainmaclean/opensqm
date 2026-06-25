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


def _build_cys_cph(pH: float, *, state_index: int = 0) -> ConstantPH:
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
        pH=pH,
        config=ConstantpHSettings(),
        references={residues[1].name: pair_reference},
        titratable_residue_indices=[1],
        ring_flip_angles=None,
    )
    cph.setResidueState(1, state_index, relax=False)
    return cph


def test_replica_exchange_probability_same_proton_count() -> None:
    """Equal N gives unit acceptance regardless of pH separation."""
    cph_low = _build_cys_cph(4.0, state_index=0)
    cph_high = _build_cys_cph(10.0, state_index=0)
    assert num_titratable_protons(cph_low) == num_titratable_protons(cph_high)
    assert replica_exchange_log_probability(cph_low, cph_high) == 0.0


def test_replica_exchange_probability_favours_matching_ph() -> None:
    """More-protonated replica favours swapping toward lower pH."""
    protonated = _build_cys_cph(4.0, state_index=0)
    deprotonated = _build_cys_cph(10.0, state_index=1)
    assert num_titratable_protons(protonated) > num_titratable_protons(deprotonated)
    log_prob = replica_exchange_log_probability(protonated, deprotonated)
    expected = (
        (num_titratable_protons(protonated) - num_titratable_protons(deprotonated))
        * math.log(10.0)
        * (10.0 - 4.0)
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
        pH=ladder,
        config=ConstantpHSettings(),
        references={residues[1].name: pair_reference},
        titratable_residue_indices=[1],
        n_replicas=1,
        ring_flip_angles=None,
    )
    assert remd.current_ph_values() == [4.0]
    assert remd.attemptAdjacentExchanges() == []


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
        pH=ladder,
        config=ConstantpHSettings(),
        references={residues[1].name: pair_reference},
        titratable_residue_indices=[1],
        n_replicas=2,
        ring_flip_angles=None,
    )
    assert remd.current_ph_values() == [4.0, 10.0]
    assert len(remd.replicas) == 2


def test_remd_swap_exchanges_ph_and_protonation() -> None:
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
        pH=ladder,
        config=ConstantpHSettings(),
        references={residues[1].name: pair_reference},
        titratable_residue_indices=[1],
        n_replicas=2,
        ring_flip_angles=None,
    )
    remd.replicas[0].setResidueState(1, 0, relax=False)
    remd.replicas[1].setResidueState(1, 1, relax=False)
    remd.replicas[0].currentPHIndex = 0
    remd.replicas[1].currentPHIndex = 1

    np.random.seed(0)
    assert remd.attemptReplicaExchange(0, 1)

    assert remd.replicas[0].titrations[1].currentIndex == 1
    assert remd.replicas[1].titrations[1].currentIndex == 0
    assert remd.replicas[0].currentPHIndex == 1
    assert remd.replicas[1].currentPHIndex == 0


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
        pH=ladder,
        config=ConstantpHSettings(),
        references=references,
        titratable_residue_indices=titratable,
        n_replicas=2,
        ring_flip_angles=None,
    )
    remd.set_weights([0.0, 1.5, 3.0])
    remd.step(25)
    remd.replicas[0].setResidueState(1, 1, relax=False)
    remd.replicas[0].currentPHIndex = 2
    remd.replicas[1].currentPHIndex = 0
    expected_steps = remd.replicas[0].simulation.currentStep

    ckpt_dir = tmp_path / "checkpoints"
    remd.save_checkpoint(ckpt_dir)

    restored = ConstantPHRemd(
        topology=pdb.topology,
        positions=pdb.positions,
        pH=ladder,
        config=ConstantpHSettings(),
        references=references,
        titratable_residue_indices=titratable,
        n_replicas=2,
        ring_flip_angles=None,
    )
    restored.load_checkpoint(ckpt_dir)

    assert restored.weights == remd.weights
    assert restored.replicas[0].simulation.currentStep == expected_steps
    assert restored.replicas[0].titrations[1].currentIndex == 1
    assert restored.replicas[0].currentPHIndex == 2
    assert restored.replicas[1].currentPHIndex == 0
