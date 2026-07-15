"""Tests for ``build_transitions_tree`` and ``_validate_transitions_graph``.

Covers the standard charge ladder, multi-child trees, and the "same-charge
tautomer" case where the variant set is missing a charge-(q+1) ancestor
for one of its highest-charge variants. In that case the offending variant
is attached as a *parent* of a charge-(q-1) sink instead of as a child of
a (non-existent) charge-(q+1) parent, so the resulting graph still spans
all variants by 1-proton steps.
"""

import pytest
from openff.toolkit.topology import Molecule  # type: ignore
from rdkit import Chem

from opensqm.cph.reference_energy import (
    Transition,
    _validate_transitions_graph,
    build_transitions_tree,
)


def _named_offmol(smiles: str, name: str) -> Molecule:
    """Helper: turn a SMILES into an OpenFF Molecule with an assigned name."""
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    offmol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
    offmol.name = name
    return offmol


def _stub_pka(parent: Chem.Mol, child: Chem.Mol) -> float:
    """A pKa stub that returns the parent->child charge drop (always 1.0).

    Lets us exercise the tree builder without depending on a real pKa
    predictor; the actual value is irrelevant to the topology of the
    transitions list returned.
    """
    return float(Chem.GetFormalCharge(parent) - Chem.GetFormalCharge(child))


def test_build_transitions_tree_single_variant() -> None:
    """A lone protomer has no protonation transitions."""
    neutral = _named_offmol("NCC(=O)O", "LIG")

    transitions = build_transitions_tree([neutral], pka_fn=_stub_pka)

    assert transitions == []


def test_build_transitions_tree_linear_ladder() -> None:
    """+1 -> 0 -> -1 forms a linear chain of two transitions."""
    cation = _named_offmol("[NH3+]CC(=O)O", "LIG")
    neutral = _named_offmol("NCC(=O)O", "LIG1")
    anion = _named_offmol("NCC(=O)[O-]", "LIG2")

    transitions = build_transitions_tree(
        [cation, neutral, anion],
        pka_fn=_stub_pka,
    )

    assert transitions == [
        ("LIG", "LIG1", 1.0),
        ("LIG1", "LIG2", 1.0),
    ]


def test_build_transitions_tree_star() -> None:
    """A +1 root with two charge-0 tautomers forms a star tree.

    ``build_transitions_tree`` only inspects ``Chem.GetFormalCharge``
    and the variant names, so we can stand in for a HIS-like residue
    with three small molecules that happen to carry the right total
    charges (the chemistry between them is irrelevant to this test).
    """
    cation = _named_offmol("[NH3+]CC", "HIP")
    neutral_a = _named_offmol("NCC", "HID")
    neutral_b = _named_offmol("OCC", "HIE")

    transitions = build_transitions_tree(
        [cation, neutral_a, neutral_b],
        pka_fn=_stub_pka,
    )

    assert transitions == [
        ("HIP", "HID", 1.0),
        ("HIP", "HIE", 1.0),
    ]


def test_build_transitions_tree_same_charge_tautomer_with_missing_parent() -> None:
    """Two -1 tautomers + one -2 anion: the second -1 attaches as a parent of -2.

    Reproduces the aspartate/glutamate-like scenario where unipka returns
    two ``-1`` microstates (zwitterion vs neutral-amine + protonated
    acid) and a single ``-2`` microstate. There is no ``0`` ancestor in
    the variant set, so the second ``-1`` variant cannot find a parent
    at charge ``+1`` higher; instead it should be attached as a parent
    of the ``-2`` sink.
    """
    zwitter = _named_offmol("[NH3+]C(CC(=O)[O-])C(=O)[O-]", "LIG")
    neutral_amine = _named_offmol("NC(CC(=O)[O-])C(=O)O", "LIG1")
    dianion = _named_offmol("NC(CC(=O)[O-])C(=O)[O-]", "LIG2")

    transitions = build_transitions_tree(
        [zwitter, neutral_amine, dianion],
        pka_fn=_stub_pka,
    )

    # LIG1 has no -1 + 1 = 0 parent so it attaches as a parent of LIG2.
    # LIG2 finds LIG (the first -1 variant) as its standard parent.
    assert transitions == [
        ("LIG1", "LIG2", 1.0),
        ("LIG", "LIG2", 1.0),
    ]


def test_build_transitions_tree_isolated_tautomer_is_rejected() -> None:
    """Two same-charge variants with nowhere to deprotonate to: error.

    With no variant at ``charge - 1`` to act as a sink and no variant
    at ``charge + 1`` to act as a parent, the second neutral cannot be
    connected to the first by a 1-proton step inside this set, so
    ``build_transitions_tree`` should refuse to fabricate an edge.
    """
    a = _named_offmol("CCO", "VAR_A")
    b = _named_offmol("CCN", "VAR_B")

    with pytest.raises(ValueError, match="Cannot place variant"):
        build_transitions_tree([a, b], pka_fn=_stub_pka)


def test_validate_transitions_graph_accepts_undirected_connectivity() -> None:
    """Edges that do not all point away from the root are accepted iff connected.

    Mirrors the build_transitions_tree output for the
    ``[-1, -1, -2]`` variant set: the second ``-1`` variant points
    *into* the same ``-2`` sink as the root. The directed graph leaves
    the second ``-1`` unreachable from the root, but the undirected
    graph is connected, which is enough for the LS reference-energy
    solver.
    """
    transitions = [
        Transition(parent=1, child=2, pka=1.0),
        Transition(parent=0, child=2, pka=1.0),
    ]

    _validate_transitions_graph(transitions, n_variants=3, root_idx=0)


def test_validate_transitions_graph_rejects_disconnected_components() -> None:
    """A variant in a component disjoint from the root must still raise.

    Three edges across four variants would *normally* be a spanning
    tree, but here the third edge piles a redundant arrow onto the
    ``{0, 1}`` component while the ``{2, 3}`` component remains
    untouched, so undirected connectivity to the root still fails.
    """
    transitions = [
        Transition(parent=0, child=1, pka=1.0),
        Transition(parent=1, child=0, pka=1.0),
        Transition(parent=2, child=3, pka=1.0),
    ]

    with pytest.raises(ValueError, match="not connected to root"):
        _validate_transitions_graph(transitions, n_variants=4, root_idx=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
