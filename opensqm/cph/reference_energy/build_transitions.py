"""High-level constructors for the transition graph that connects variants.

:func:`build_transitions_tree` is the user-facing entry point: feed it a
list of OpenFF :class:`Molecule` protomers and a pKa-prediction callback
and it returns a graph of :data:`opensqm.cph.reference_energy.types.NamedTransition`
edges suitable for :func:`opensqm.cph.reference_energy.generate.generate_ligand_reference`.

:func:`_resolve_named_transitions` is the internal counterpart that
turns those name-based tuples into index-based
:class:`opensqm.cph.reference_energy.models.Transition` objects once the
variant list is locked in.
"""
# pyrefly: ignore [missing-import]
from openff.toolkit.topology import Molecule  # type: ignore
from rdkit import Chem

from .models import Transition
from .types import NamedTransition


def build_transitions_tree(
    variant_molecules: list[Molecule],
    pka_fn,
) -> list[NamedTransition]:
    """Build a 1-proton-step spanning graph of microscopic transitions.

    Walks the variants from highest to lowest formal charge and, for each
    non-root variant ``v``, picks one of two attachment strategies:

    * **Standard deprotonation child.** If some earlier variant has
      charge ``charge[v] + 1``, attach ``v`` as the child of the first
      such variant. This is the only strategy used when the input is a
      strict, gap-free charge ladder (e.g. ``+1 -> 0 -> -1``).
    * **Same-charge tautomer.** If ``v`` has no earlier variant at
      ``charge[v] + 1`` but does share its charge with an earlier
      variant, ``v`` is itself a tautomer at the highest charge level
      represented in ``variant_molecules``. There is no valid 1-proton
      "parent" for it inside the variant set, so instead attach ``v``
      *as a parent* of the first variant at ``charge[v] - 1``. Both the
      root and ``v`` then point at the same lower-charge sink, which is
      enough for the least-squares reference-energy solver to determine
      ``v``'s reference energy relative to the root via
      ``E_root - E_sink`` and ``E_v - E_sink`` constraints.

    Variants are processed in the order they appear in
    ``variant_molecules``, so within a given charge level the
    earliest-listed variant is treated as the "best" parent / sink.
    With unipka inputs, sorting by ``("charge" desc, "free energy" asc)``
    therefore makes the most-stable tautomer at each charge level the
    canonical parent.

    Parameters
    ----------
    variant_molecules : list[openff.toolkit.topology.Molecule]
        Protonation states of the residue/ligand. Must be ordered such
        that any variant at charge ``q`` precedes every variant at charge
        ``q - 1`` -- e.g. by sorting a unipka microstate distribution by
        ``charge`` descending. Each molecule's ``.name`` is used as the
        transition's parent/child label.
    pka_fn : Callable[[Chem.Mol, Chem.Mol], float]
        Returns the microscopic pKa of the parent -> child deprotonation
        for a given pair of RDKit molecules. For a unipka distribution
        ``pka_fn = lambda p, c: unipka.get_macro_pka_from_macrostates(
        acid_macrostate=[p], base_macrostate=[c])`` works directly:
        with one tautomer per microstate, unipka's "macro" pKa equals the
        microscopic value.

    Returns
    -------
    list[NamedTransition]
        Up to ``len(variant_molecules) - 1`` ``(parent_name, child_name, pka)``
        tuples whose underlying undirected graph spans
        ``variant_molecules`` from ``variant_molecules[0]``. Each edge
        drops the formal charge by exactly one. Returns an empty list when
        ``variant_molecules`` contains a single protomer (no protonation
        transitions to fit). Suitable for direct use as the second element
        of ``ligands`` entries passed to :func:`generate_all` or as the
        ``transitions`` argument of :func:`generate_ligand_reference`.

    Raises
    ------
    ValueError
        If a non-root variant cannot be placed under either strategy
        above (e.g. an isolated tautomer with no charge-(q-1) sink, or a
        charge-ladder gap that leaves some variant disconnected).
    """
    if len(variant_molecules) < 2:
        return []
    rdkit_mols = [m.to_rdkit() for m in variant_molecules]
    charges = [int(Chem.GetFormalCharge(rd)) for rd in rdkit_mols]
    transitions: list[NamedTransition] = []
    for var_idx in range(1, len(variant_molecules)):
        target_parent_charge = charges[var_idx] + 1
        parent_idx = next(
            (i for i in range(var_idx) if charges[i] == target_parent_charge),
            None,
        )
        if parent_idx is not None:
            pka = pka_fn(rdkit_mols[parent_idx], rdkit_mols[var_idx])
            transitions.append(
                (
                    variant_molecules[parent_idx].name,
                    variant_molecules[var_idx].name,
                    float(pka),
                )
            )
            continue

        # No earlier variant at charge+1: this is a same-charge tautomer
        # of the root (or another earlier variant). Attach it as a parent
        # of the first variant at charge-1 instead.
        same_charge_earlier = any(
            charges[i] == charges[var_idx] for i in range(var_idx)
        )
        if same_charge_earlier:
            target_child_charge = charges[var_idx] - 1
            child_idx = next(
                (
                    i
                    for i in range(len(variant_molecules))
                    if i != var_idx and charges[i] == target_child_charge
                ),
                None,
            )
            if child_idx is not None:
                pka = pka_fn(rdkit_mols[var_idx], rdkit_mols[child_idx])
                transitions.append(
                    (
                        variant_molecules[var_idx].name,
                        variant_molecules[child_idx].name,
                        float(pka),
                    )
                )
                continue

        available = sorted(set(charges[:var_idx]), reverse=True)
        raise ValueError(
            f"Cannot place variant {variant_molecules[var_idx].name!r} "
            f"(charge {charges[var_idx]}) on the transition tree: no "
            f"earlier variant has charge {target_parent_charge}, and no "
            f"variant at charge {charges[var_idx] - 1} is available to "
            f"act as a deprotonation sink either. Earlier-variant "
            f"charges: {available}. Make sure the variant set is "
            f"connected by 1-proton steps and sorted by charge descending."
        )
    return transitions


def _resolve_named_transitions(
    transitions: list[NamedTransition] | list[tuple],
    names: list[str],
) -> list[Transition]:
    """Convert ``(parent_name, child_name, pka)`` tuples to ``Transition`` objects.

    ``names`` is the variant-name list (e.g. ``['HIP', 'HID', 'HIE']`` for
    proteins, or the per-variant ``Molecule.name`` strings for ligands).
    Names must be unique within ``names``.
    """
    name_to_idx: dict[str, int] = {}
    for i, n in enumerate(names):
        if n in name_to_idx:
            raise ValueError(
                f"variant names must be unique to resolve transitions; "
                f"got duplicate {n!r}"
            )
        name_to_idx[n] = i
    out: list[Transition] = []
    for entry in transitions:
        if len(entry) != 3:
            raise ValueError(
                f"each transition must be a (parent, child, pka) tuple; got {entry!r}"
            )
        parent_name, child_name, pka = entry
        if parent_name not in name_to_idx:
            raise ValueError(
                f"transition parent {parent_name!r} is not one of the variants {names!r}"
            )
        if child_name not in name_to_idx:
            raise ValueError(
                f"transition child {child_name!r} is not one of the variants {names!r}"
            )
        out.append(
            Transition(
                parent=name_to_idx[parent_name],
                child=name_to_idx[child_name],
                pka=float(pka),
            )
        )
    return out


__all__ = ["build_transitions_tree", "_resolve_named_transitions"]
