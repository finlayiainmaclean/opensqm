"""Graph-level operations on a list of microscopic ``Transition`` edges.

Everything here is a *pure* function of the transitions themselves --
no force-field, no MD context, no Pydantic models -- so the heavier
modules in the package (``models``, ``finder``, ``generate``) can lean
on these helpers without worrying about import cycles. The ``Transition``
type is referenced via :pep:`0563` forward references; the runtime body
of every function only touches the ``.parent``, ``.child`` and ``.pka``
attributes, so we don't need to import the concrete class.

The fitting model used throughout the package is:

    E_ref[child] - E_ref[parent] = ΔE_pair    (one constraint per edge)
    E_ref[root]                  = 0           (gauge fix)

with ΔE_pair coming from a pairwise reference-energy MC measurement at a
single edge. :func:`_solve_reference_energies_ls` solves this linear
system in least-squares form so cycle-redundant inputs degrade
gracefully into a residual report instead of an inconsistency error.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .models import Transition


def macro_pka(micro_pkas) -> float:
    """Combine a list of micro pKas into a single macroscopic pKa.

    For a residue that can deprotonate via multiple parallel pathways from a
    common protonated state (e.g. HIP -> HID and HIP -> HIE), the macroscopic
    dissociation constant is the sum of the individual microscopic constants:

        10**(-pKa_macro) = sum_i 10**(-pKa_i)

    Parameters
    ----------
    micro_pkas: iterable of float
        The microscopic pKa values for each parallel deprotonation pathway.

    Returns
    -------
    float
        The macroscopic pKa.
    """
    micro_pkas = np.asarray(list(micro_pkas), dtype=float)
    if micro_pkas.size == 0:
        raise ValueError("micro_pkas must contain at least one value")
    return -np.log10(np.sum(10.0**(-micro_pkas)))


def _validate_transitions_graph(
    transitions: list["Transition"],
    n_variants: int,
    root_idx: int = 0,
) -> None:
    """Validate that ``transitions`` connect every variant to ``root_idx``.

    Each transition is a directed parent->child edge that physically
    represents a deprotonation step (parent_charge - child_charge == 1).
    For the least-squares reference-energy fit to be solvable, the
    *undirected* graph induced by these edges must be connected -- i.e.
    every variant must lie in the same component as ``root_idx``. The
    parent->child arrows themselves do not need to all point away from
    the root: when the input variant set contains tautomers at the same
    charge level (e.g. two ``-1`` protomers that share a missing ``0``
    ancestor) some edges naturally point "into" the root's charge level
    instead of away from it. The LS solver in
    :func:`_solve_reference_energies_ls` treats each edge as an
    undirected linear constraint and only cares that the edges span all
    variants.

    A spanning tree (``len(transitions) == n_variants - 1``) is the
    unique-solution minimum; redundant edges
    (``len(transitions) > n_variants - 1``) create cycles whose
    residuals are absorbed by the LS fit.

    Raises ``ValueError`` if the structure is malformed (out-of-range
    indices, self-loops, or the undirected graph fails to span all
    variants).
    """
    if n_variants <= 1:
        if transitions:
            raise ValueError(
                f"single-variant residue must have no transitions; "
                f"got {len(transitions)}"
            )
        return
    if len(transitions) < n_variants - 1:
        raise ValueError(
            f"transitions must span all {n_variants} variants -- need at "
            f"least n_variants - 1 = {n_variants - 1} edges, got "
            f"{len(transitions)}"
        )
    neighbors_of: dict[int, list[int]] = defaultdict(list)
    for t in transitions:
        if not (0 <= t.parent < n_variants and 0 <= t.child < n_variants):
            raise ValueError(
                f"transition indices out of range for {n_variants} variants: {t}"
            )
        if t.parent == t.child:
            raise ValueError(f"self-loop in transitions: {t}")
        neighbors_of[t.parent].append(t.child)
        neighbors_of[t.child].append(t.parent)
    visited = {root_idx}
    queue = deque([root_idx])
    while queue:
        cur = queue.popleft()
        for nbr in neighbors_of.get(cur, ()):
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    if len(visited) != n_variants:
        unreached = sorted(set(range(n_variants)) - visited)
        raise ValueError(
            f"variants not connected to root {root_idx}: {unreached}. "
            f"Make sure the transitions span the full charge ladder."
        )


def _solve_reference_energies_ls(
    transitions: list["Transition"],
    measured_deltas_kj: list[float],
    n_variants: int,
    root_idx: int = 0,
) -> tuple[list[float], list[float]]:
    """Linear-least-squares fit of per-variant reference energies from edge data.

    Each transition contributes a constraint
    ``E_ref[child] - E_ref[parent] ≈ measured_deltas_kj[edge]``, with the
    root pinned to ``E_ref[root_idx] = 0``. For a spanning tree
    (``len(transitions) == n_variants - 1``) the system has a unique
    exact solution and this routine returns the same numbers as a naive
    walk of the tree. Redundant edges (``len(transitions) > n_variants - 1``)
    over-determine the system: the LS fit averages the closure errors
    across the cycle and the per-edge residuals
    ``measured - (E_child - E_parent)`` quantify how far the experimental
    inputs are from being thermodynamically self-consistent.

    Parameters
    ----------
    transitions : list[Transition]
        One entry per measured edge; ordering does not matter.
    measured_deltas_kj : list[float]
        Same length as ``transitions``; the measured per-edge
        ``E_ref[child] - E_ref[parent]`` in kJ/mol from the pairwise
        finder.
    n_variants : int
        Total number of variants.
    root_idx : int, default 0
        Index of the variant whose reference energy is pinned to 0
        (typically ``main_variant``).

    Returns
    -------
    tuple[list[float], list[float]]
        ``(reference_energies_kj, per_edge_residuals_kj)``. The first
        entry has length ``n_variants`` with
        ``reference_energies_kj[root_idx] == 0``; the second has length
        ``len(transitions)`` and gives ``measured - (E_child - E_parent)``
        for each edge in the same order as ``transitions``.
    """
    if len(transitions) != len(measured_deltas_kj):
        raise ValueError(
            "transitions and measured_deltas_kj must have the same length"
        )
    free_idx = [i for i in range(n_variants) if i != root_idx]
    if not free_idx:
        return ([0.0] * n_variants, [])
    col_of = {idx: k for k, idx in enumerate(free_idx)}
    n_edges, n_unknowns = len(transitions), len(free_idx)
    a_matrix = np.zeros((n_edges, n_unknowns))
    b_vec = np.zeros(n_edges)
    for row, (t, delta_kj) in enumerate(
        zip(transitions, measured_deltas_kj, strict=True),
    ):
        if t.parent != root_idx:
            a_matrix[row, col_of[t.parent]] = -1.0
        if t.child != root_idx:
            a_matrix[row, col_of[t.child]] = 1.0
        b_vec[row] = delta_kj
    x, _residuals_sumsq, rank, _sv = np.linalg.lstsq(a_matrix, b_vec, rcond=None)
    if rank < n_unknowns:
        raise ValueError(
            f"transitions are under-determined (LS rank {rank} < {n_unknowns} "
            f"unknowns); some variants are not reachable from root {root_idx} "
            f"via the supplied edges"
        )
    edge_residuals = b_vec - a_matrix @ x
    energies = [0.0] * n_variants
    for idx, k in col_of.items():
        energies[idx] = float(x[k])
    return energies, [float(r) for r in edge_residuals]


# Back-compat alias: callers from earlier in this refactor still use the
# tree-only name. The new graph validator is a strict superset (it accepts
# trees and rejects them only when they fail to span), so this redirect is
# safe.
_validate_transitions_tree = _validate_transitions_graph



def _topological_transitions(
    transitions: list["Transition"],
    root_idx: int = 0,
) -> list["Transition"]:
    """Return ``transitions`` BFS-ordered from ``root_idx`` for diagnostics.

    Mainly a logging convenience: with the LS-based reference-energy fit
    the order in which the per-edge ΔE measurements are computed does not
    matter, but printing them parent-first reads more naturally for users.
    For graphs with cycles a single deterministic order is still produced
    (each edge is yielded the first time its parent is dequeued).
    """
    children_of: dict[int, list["Transition"]] = defaultdict(list)
    for t in transitions:
        children_of[t.parent].append(t)
    ordered: list["Transition"] = []
    visited_edges: set[int] = set()
    queue = deque([root_idx])
    seen_nodes = {root_idx}
    while queue:
        cur = queue.popleft()
        for t in children_of.get(cur, ()):
            edge_id = id(t)
            if edge_id in visited_edges:
                continue
            visited_edges.add(edge_id)
            ordered.append(t)
            if t.child not in seen_nodes:
                seen_nodes.add(t.child)
                queue.append(t.child)
    # Append any edges not reached by BFS (defensive: caller should have
    # validated reachability already).
    for t in transitions:
        if id(t) not in visited_edges:
            ordered.append(t)
    return ordered


__all__ = [
    "_solve_reference_energies_ls",
    "_topological_transitions",
    "_validate_transitions_graph",
    "_validate_transitions_tree",
    "macro_pka",
]
