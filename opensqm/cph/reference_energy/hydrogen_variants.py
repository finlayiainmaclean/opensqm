"""Read a per-residue ``Modeller.addHydrogens`` variant list off a topology.

Used by :func:`opensqm.cph.reference_energy.generate.generate_ligand_reference`
to capture each protomer's hydrogen layout once the OpenFF
:class:`Molecule` has been converted to an OpenMM topology, and by the
``test_get_hydrogen_variants`` regression tests that round-trip through
``addHydrogens`` to confirm the variant list reproduces the input.
"""

# pyrefly: ignore [missing-import]
from collections import defaultdict

from openmm.app import Topology
from openmm.app import element as elem


def get_hydrogen_variants(topology: Topology) -> list[list[tuple[str, str]] | None]:
    """Given an OpenMM Topology, return a per-residue list of (hydrogen_name, parent_name) tuples.

    This produces output in the format expected by Modeller.addHydrogens(variants=...).
    """
    # Build per-residue hydrogen -> parent mapping from bonds
    residue_hydrogens = defaultdict(list)
    for bond in topology.bonds():
        a1, a2 = bond[0], bond[1]
        if a1.element == elem.hydrogen and a1.residue == a2.residue:
            residue_hydrogens[a1.residue.index].append((a1.name, a2.name))
        elif a2.element == elem.hydrogen and a1.residue == a2.residue:
            residue_hydrogens[a2.residue.index].append((a2.name, a1.name))

    # Build the variants list (one entry per residue)
    variants: list[list[tuple[str, str]] | None] = []
    for residue in topology.residues():
        if residue.index in residue_hydrogens:
            variants.append(residue_hydrogens[residue.index])
        else:
            variants.append(None)
    return variants


__all__ = ["get_hydrogen_variants"]
