"""Pydantic data models for titratable-residue reference data.

:class:`Transition` is the atomic unit -- a single ``parent -> child``
deprotonation edge with its microscopic pKa -- and
:class:`TitratableResidueReference` aggregates per-residue variants,
charges, energies and the connecting transition graph into one
serialisable object that the rest of the constant-pH pipeline reads from
disk.

The graph-level invariants (undirected connectivity from the root, edges
that drop charge by exactly 1, etc.) are enforced by Pydantic root
validators that delegate to the helpers in
:mod:`opensqm.cph.reference_energy.graph`.
"""
# pyrefly: ignore [missing-import]
from collections import defaultdict
from pathlib import Path

from openmm.unit import kilojoules_per_mole
from pydantic import BaseModel, Field, root_validator

from opensqm.md.terminal_ring_mc import RING_FLIP_BONDS

from .graph import _validate_transitions_graph, macro_pka
from .model_compounds import _PROTEIN_VARIANT_CHARGES
from .types import VariantSpec


class Transition(BaseModel):
    """A 1-proton microscopic deprotonation step in a titration tree.

    The ``parent`` and ``child`` indices reference positions in the
    surrounding ``TitratableResidueReference.variants`` list. ``parent`` is
    the more-protonated state, ``child`` the less-protonated state, and the
    proton count drops by exactly one across the edge.
    """

    parent: int
    child: int
    pka: float

    def as_tuple(self) -> tuple[int, int, float]:
        """Return ``(parent, child, pka)`` as a plain Python tuple."""
        return (self.parent, self.child, float(self.pka))


class TitratableResidueReference(BaseModel):
    """Reference data for one titratable residue type at a fixed simulation config.

    A titratable residue (e.g. histidine, or a small ligand) can adopt
    several protonation "variants" connected by a tree of microscopic
    1-proton transitions. One variant is chosen as ``main_variant`` (the
    tree root) and all other quantities are expressed relative to it.

    Two flavours of variant entry are supported, mirroring what
    ``Modeller.addHydrogens(variants=...)`` accepts:

    * **String variant names** (e.g. ``"HIP"``, ``"HID"``, ``"HIE"``) for
      residues that are described in OpenMM's default ``Hydrogens.xml``
      database. ``variants[0]`` must equal ``main_variant`` in this case.
    * **Per-state hydrogen layouts** — a list of
      ``(hydrogen_name, parent_heavy_atom_name)`` tuples — for custom
      residues such as ligands, where the topology residue name itself
      (e.g. ``"LIG"``) plays the role of ``main_variant``.

    All variant entries within a single object must be of the same flavour.

    Attributes
    ----------
    residue_name : str
        PDB residue name as it appears in the input topology (e.g. ``"HIS"``
        for a protein histidine, or ``"LIG"`` for a custom imidazolium-like
        ligand).
    main_variant : str
        For protein residues, the variant against which the others are
        referenced (e.g. ``"HIP"`` for HIS); must equal ``variants[0]``.
        For custom ligands, the topology residue name of the main
        protonation state (the one that the production topology is built
        around).
    variant_names : list[str]
        Human-readable name of each variant, same length and order as
        ``variants``. For protein residues this is identical to
        ``variants`` (e.g. ``["HIP", "HID", "HIE"]``); for custom ligands
        it carries the per-state :attr:`openff.toolkit.topology.Molecule.name`
        (e.g. ``["LIG", "LIG1", "LIG2"]``), which are otherwise lost when
        ``variants`` is a list of per-state hydrogen layouts. Used to
        label columns in population summaries and to format macro-pKa
        output without requiring callers to thread the names through
        separately.
    variants : list[VariantSpec]
        All variants this residue can adopt during the simulation, with the
        main variant first. Suitable for direct use as the value side of
        ``ConstantPH``'s ``residueVariants`` mapping.
    charges : list[int]
        Formal charge of each variant (e.g. ``[1, 0, 0]`` for HIS's
        ``[HIP, HID, HIE]``). Same length as ``variants``. Used by
        ``ConstantPH`` to label states by macroscopic charge and by macro
        pKa post-processing in the run loop.
    reference_energies_kj_per_mole : list[float]
        Reference energy of each variant, in kJ/mol, relative to the main
        variant. Same length as ``variants``; the first entry is ``0.0`` by
        construction.
    transitions : list[Transition]
        Microscopic 1-proton transitions that span the variants graph from
        the root (``variants[0]``). For HIS this is the star tree
        ``[(HIP -> HID, 7.1), (HIP -> HIE, 6.5)]``; deeper ladders such as
        ``HIP -> HID -> HIN`` add another edge ``(HID -> HIN, 14.0)``.
        Redundant edges that close cycles (e.g. both ``HID -> HIN`` and
        ``HIE -> HIN``) are allowed; the per-state reference energies are
        then obtained from a least-squares fit over the edges and the
        cycle-closure residuals are logged for diagnostic purposes.
        Length must be at least ``len(variants) - 1`` and the
        *undirected* graph induced by the edges must connect every
        variant to the root. The directed parent->child arrows
        themselves do not need to all point away from the root: e.g.
        when the variant set contains tautomers at the same charge level
        (such as two ``-1`` ligand protomers that share a missing
        charge-``0`` ancestor) some edges naturally point toward the
        root's charge level instead of away from it.
    ring_flip_bonds : list[tuple[str, str]]
        Heavy-atom bonds whose 180-degree rotation should be proposed as
        terminal-group MC moves by ``ConstantPH``. Each entry is a
        ``(anchor_atom_name, pivot_atom_name)`` pair naming the two ends
        of the rotatable bond; the rotatable side is determined
        automatically by :func:`opensqm.md.terminal_ring_mc.find_terminal_group`
        via a graph split on the bond. For HIS this is ``[("CB", "CG")]``;
        for ligands the bonds are typically discovered via
        :func:`opensqm.torsion_scanner.autodetect_flip_dihedrals` (after
        translating rdkit atom indices to topology atom names). An empty
        list disables ring-flip MC for residues of this type. Atom names
        must exist in every variant's topology (heavy atoms are stable
        across protonation states, so a single bond suffices for all
        variants).
    """

    residue_name: str
    main_variant: str
    variant_names: list[str]
    variants: list[VariantSpec]
    charges: list[int]
    reference_energies_kj_per_mole: list[float]
    transitions: list[Transition]
    ring_flip_bonds: list[tuple[str, str]] = Field(default_factory=list)

    @root_validator(pre=True)
    def _migrate_legacy(cls, values):  # noqa: N805
        """Backfill ``transitions``/``charges`` from the legacy ``micro_pkas`` schema.

        Earlier versions of this file stored a flat ``micro_pkas`` list of
        length ``len(variants) - 1``, indexed against ``main_variant`` (i.e.
        a star tree). Cached JSON files written under that schema are
        migrated in-place when loaded so the user does not have to manually
        re-run the reference-energy fits.
        """
        if not isinstance(values, dict):
            return values
        values = dict(values)
        if "transitions" not in values and "micro_pkas" in values:
            variants = values.get("variants") or []
            micro_pkas = values.get("micro_pkas") or []
            if len(micro_pkas) != len(variants) - 1:
                raise ValueError(
                    "legacy micro_pkas length must equal len(variants) - 1"
                )
            values["transitions"] = [
                {"parent": 0, "child": i + 1, "pka": float(pka)}
                for i, pka in enumerate(micro_pkas)
            ]
        values.pop("micro_pkas", None)
        if "charges" not in values:
            variants = values.get("variants") or []
            if all(isinstance(v, str) for v in variants):
                try:
                    values["charges"] = [
                        _PROTEIN_VARIANT_CHARGES[v] for v in variants
                    ]
                except KeyError as exc:
                    raise ValueError(
                        f"legacy reference uses unknown protein variant {exc!r}; "
                        f"add it to _PROTEIN_VARIANT_CHARGES or regenerate"
                    ) from exc
            else:
                raise ValueError(
                    "legacy ligand reference is missing `charges` and they "
                    "cannot be inferred from variant hydrogen layouts; delete "
                    "the cache file to regenerate."
                )
        if "variant_names" not in values:
            variants = values.get("variants") or []
            main_variant = values.get("main_variant")
            if variants and all(isinstance(v, str) for v in variants):
                # Protein residues: the variant name *is* the string entry.
                values["variant_names"] = list(variants)
            elif variants and main_variant:
                # Legacy ligand caches predate ``variant_names``. Reconstruct
                # the run.py convention (``main`` for state 0, then
                # ``main`` suffixed with the state index) so cached files
                # remain loadable without forcing a full regenerate.
                values["variant_names"] = [
                    main_variant if i == 0 else f"{main_variant}{i}"
                    for i in range(len(variants))
                ]
        if "ring_flip_bonds" not in values:
            # Legacy caches (protein + ligand) predate ``ring_flip_bonds``.
            # Backfill the protein default from ``RING_FLIP_BONDS`` so e.g.
            # HIS keeps its CB-CG flip without forcing a regenerate; ligand
            # caches default to "no flips" (callers needing them must
            # regenerate, since flip discovery is rdkit-side and not
            # reconstructible from the cached reference alone).
            residue_name = values.get("residue_name")
            if residue_name and residue_name.upper() in RING_FLIP_BONDS:
                values["ring_flip_bonds"] = [tuple(RING_FLIP_BONDS[residue_name.upper()])]
            else:
                values["ring_flip_bonds"] = []
        return values

    @root_validator
    def _check_consistency(cls, values):  # noqa: N805
        variants = values.get("variants") or []
        variant_names = values.get("variant_names") or []
        main_variant = values.get("main_variant")
        energies = values.get("reference_energies_kj_per_mole") or []
        charges = values.get("charges") or []
        transitions = values.get("transitions") or []
        if not variants:
            raise ValueError("variants must not be empty")
        first_is_str = isinstance(variants[0], str)
        if any(isinstance(v, str) != first_is_str for v in variants):
            raise ValueError(
                "variants must be homogeneous: either all string variant names "
                "or all per-state hydrogen layouts"
            )
        if first_is_str and variants[0] != main_variant:
            raise ValueError(
                f"variants[0] ({variants[0]!r}) must equal "
                f"main_variant ({main_variant!r})"
            )
        if len(variant_names) != len(variants):
            raise ValueError(
                "variant_names must have the same length as variants"
            )
        if len(set(variant_names)) != len(variant_names):
            raise ValueError(
                f"variant_names must be unique, got {variant_names!r}"
            )
        if variant_names[0] != main_variant:
            raise ValueError(
                f"variant_names[0] ({variant_names[0]!r}) must equal "
                f"main_variant ({main_variant!r})"
            )
        if first_is_str and variant_names != variants:
            raise ValueError(
                "for string-typed variants, variant_names must match variants "
                f"exactly; got variant_names={variant_names!r} "
                f"variants={variants!r}"
            )
        if len(energies) != len(variants):
            raise ValueError(
                "reference_energies_kj_per_mole must have the same length as variants"
            )
        if energies[0] != 0.0:
            raise ValueError(
                "reference_energies_kj_per_mole[0] must be 0.0 (main_variant baseline)"
            )
        if len(charges) != len(variants):
            raise ValueError(
                "charges must have the same length as variants"
            )
        _validate_transitions_graph(transitions, len(variants), root_idx=0)
        for t in transitions:
            charge_drop = charges[t.parent] - charges[t.child]
            if charge_drop != 1:
                raise ValueError(
                    f"transition {t} drops charge by {charge_drop} (expected 1); "
                    f"each microscopic transition must remove exactly one proton"
                )
        ring_flip_bonds = values.get("ring_flip_bonds") or []
        for entry in ring_flip_bonds:
            if (
                not isinstance(entry, (tuple, list))
                or len(entry) != 2
                or not all(isinstance(n, str) and n for n in entry)
            ):
                raise ValueError(
                    f"ring_flip_bonds entries must be (anchor_name, pivot_name) "
                    f"pairs of non-empty strings, got {entry!r}"
                )
            if entry[0] == entry[1]:
                raise ValueError(
                    f"ring_flip_bonds entry {entry!r} has equal anchor and pivot "
                    f"atom names; the rotation axis would be degenerate"
                )
        return values

    @property
    def reference_energies(self) -> list:
        """Reference energies as a list of OpenMM ``Quantity`` (kJ/mol)."""
        return [v * kilojoules_per_mole for v in self.reference_energies_kj_per_mole]

    @property
    def macro_pkas_by_charge_transition(self) -> dict[tuple[int, int], float]:
        """Macroscopic pKa for each (parent_charge, child_charge) pair.

        Microscopic transitions are grouped by the formal-charge change
        they implement and the parallel paths within each group are
        combined via ``10**(-pKa_macro) = sum_i 10**(-pKa_i)``. For HIS
        this returns ``{(1, 0): macro_pka([7.1, 6.5])}``; for an extended
        HIP/HID/HIE/HIN ladder it returns the combined ``(1, 0)`` and
        ``(0, -1)`` macroscopic pKas.
        """
        by_pair: dict[tuple[int, int], list[float]] = defaultdict(list)
        for t in self.transitions:
            pair = (self.charges[t.parent], self.charges[t.child])
            by_pair[pair].append(t.pka)
        return {pair: float(macro_pka(pkas)) for pair, pkas in by_pair.items()}

    @property
    def macro_pka(self) -> float:
        """Macro pKa of the first deprotonation step away from the main variant.

        Equivalent to looking up
        ``macro_pkas_by_charge_transition[(charges[0], charges[0] - 1)]``.
        Kept for backward compatibility; for ladders deeper than one step,
        prefer ``macro_pkas_by_charge_transition`` directly.
        """
        first_pair = (self.charges[0], self.charges[0] - 1)
        return self.macro_pkas_by_charge_transition[first_pair]

    def save(self, path: Path) -> None:
        """Write the reference data to ``path`` as JSON."""
        Path(path).write_text(self.json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "TitratableResidueReference":
        """Load reference data previously written via :meth:`save`.

        Transparently migrates legacy JSON files that still use the old
        ``micro_pkas`` schema; see :meth:`_migrate_legacy`.
        """
        return cls.parse_raw(Path(path).read_text())


__all__ = ["TitratableResidueReference", "Transition"]
