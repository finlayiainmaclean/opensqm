"""Plain-Python type aliases shared across the ``reference_energy`` package.

Lifted out of the original ``reference_energy.py`` module so every other
sub-module can import these aliases without dragging in OpenMM, RDKit, or
Pydantic at module-load time.
"""
from typing import Union

# A single state's worth of hydrogen placement instructions, in the form
# expected by ``Modeller.addHydrogens(variants=...)`` for residues that are
# not described in OpenMM's default ``Hydrogens.xml``: a list of
# ``(hydrogen_name, parent_heavy_atom_name)`` pairs.
HydrogenVariant = list[tuple[str, str]]
# A variant entry for a single state, as accepted by ConstantPH/Modeller:
# either a string (the variant name in ``Hydrogens.xml``, e.g. ``"HIP"``) or
# an explicit hydrogen layout for that state.
VariantSpec = Union[str, HydrogenVariant]
# A user-facing microscopic transition: ``(parent_name, child_name, micro_pka)``.
# ``parent_name`` is the more-protonated state's variant name (for protein
# variants in ``MODEL_COMPOUNDS``) or the ``Molecule.name`` of the more-
# protonated variant (for ligand inputs). ``child_name`` is the less-
# protonated counterpart. Each transition removes exactly one proton.
NamedTransition = tuple[str, str, float]


__all__ = ["HydrogenVariant", "NamedTransition", "VariantSpec"]
