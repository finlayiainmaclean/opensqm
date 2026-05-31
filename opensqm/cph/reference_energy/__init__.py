"""Reference-energy machinery for OpenMM-CpH constant-pH simulations.

This package replaces the former monolithic ``reference_energy.py``
module; it is split across thematic sub-modules but the public surface
is unchanged. Existing callers can keep doing ::

    from opensqm.cph.reference_energy import (
        ReferenceEnergyFinder,
        Transition,
        TitratableResidueReference,
        build_protonation_states,
        build_protonation_template,
        build_transitions_tree,
        generate_all,
        generate_ligand_reference,
        get_hydrogen_variants,
        macro_pka,
        MODEL_COMPOUNDS,
    )

The sub-modules are organised roughly bottom-up, in dependency order:

* :mod:`opensqm.cph.reference_energy.types` -- plain type aliases.
* :mod:`opensqm.cph.reference_energy.model_compounds` --
  ``MODEL_COMPOUNDS`` and protein variant-charge lookups.
* :mod:`opensqm.cph.reference_energy.graph` -- transition-graph
  validation, LS solve, BFS ordering, residual logging, ``macro_pka``.
* :mod:`opensqm.cph.reference_energy.models` -- ``Transition`` and
  ``TitratableResidueReference`` Pydantic models.
* :mod:`opensqm.cph.reference_energy.build_transitions` --
  ``build_transitions_tree`` and the named-transition resolver.
* :mod:`opensqm.cph.reference_energy.protonation_states` -- aligned
  protomer construction utilities.
* :mod:`opensqm.cph.reference_energy.hydrogen_variants` --
  ``get_hydrogen_variants``.
* :mod:`opensqm.cph.reference_energy.finder` -- the iterative
  ``ReferenceEnergyFinder`` MC engine.
* :mod:`opensqm.cph.reference_energy.generate` -- top-level
  ``generate_all`` / ``generate_ligand_reference`` drivers.
"""
# pyrefly: ignore [missing-import]

# Older pickled ConstantPH/torch caches were saved before
# torch.serialization.add_safe_globals existed; provide a no-op stub so
# imports of those caches from newer torch builds don't blow up.
import torch.serialization as _torch_serialization
if not hasattr(_torch_serialization, "add_safe_globals"):
    _torch_serialization.add_safe_globals = lambda *args, **kwargs: None

from .build_transitions import _resolve_named_transitions, build_transitions_tree
from .finder import ReferenceEnergyFinder, _compute_pairwise_reference_energy
from .generate import generate_residue_reference_dict, generate_ligand_reference
from .graph import (
    _log_cycle_residuals,
    _solve_reference_energies_ls,
    _topological_transitions,
    _validate_transitions_graph,
    _validate_transitions_tree,
    macro_pka,
)
from .hydrogen_variants import get_hydrogen_variants
from .model_compounds import MODEL_COMPOUNDS, _PROTEIN_VARIANT_CHARGES
from .models import TitratableResidueReference, Transition
from .protonation_states import (
    build_protonation_states,
    build_protonation_template,
)
from .types import HydrogenVariant, NamedTransition, VariantSpec


__all__ = [
    # Public API consumed by run.py / tests / other modules.
    "ReferenceEnergyFinder",
    "Transition",
    "TitratableResidueReference",
    "MODEL_COMPOUNDS",
    "build_protonation_template",
    "build_protonation_states",
    "build_transitions_tree",
    "generate_residue_reference_dict",
    "generate_ligand_reference",
    "get_hydrogen_variants",
    "macro_pka",
    # Type aliases.
    "HydrogenVariant",
    "NamedTransition",
    "VariantSpec",
    # Internals exposed for tests and intra-package callers.
    "_PROTEIN_VARIANT_CHARGES",
    "_compute_pairwise_reference_energy",
    "_log_cycle_residuals",
    "_resolve_named_transitions",
    "_solve_reference_energies_ls",
    "_topological_transitions",
    "_validate_transitions_graph",
    "_validate_transitions_tree",
]
