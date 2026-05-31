"""High-level generation of cached :class:`TitratableResidueReference` objects.

Two entry points live here:

* :func:`generate_ligand_reference` -- run (or load) reference-energy
  fits for a single titratable ligand, given an aligned list of
  protomers (e.g. the output of
  :func:`opensqm.cph.reference_energy.protonation_states.build_protonation_states`)
  and a transitions graph (e.g. from
  :func:`opensqm.cph.reference_energy.build_transitions.build_transitions_tree`).
* :func:`generate_all` -- the top-level driver. Iterates over every
  :data:`MODEL_COMPOUNDS` entry, then optionally over each ``ligands``
  entry, and returns a ``{residue_name: TitratableResidueReference}``
  dict ready to feed back into ``ConstantPH``.

Both routines hash the ``SimulationConfig`` (forcefields, ligand
charge method, temperature, ...) into the cache filename so that
config-incompatible reference fits never collide on disk.
"""
from pathlib import Path
import xxhash
import numpy as np
from loguru import logger
from openff.toolkit.topology import Molecule  # type: ignore
from rdkit import Chem

from opensqm.cph.constantph import ConstantPH
from opensqm.cph.inchi import to_inchikey_non_standard
from opensqm.cph.simulation_config import ConstantpHSettings
from opensqm.md.prepare import solvate_ligand

from .build_transitions import _resolve_named_transitions
from .finder import (
    ReferenceEnergyFinder,
    _compute_pairwise_reference_energy,
    _make_pair_reference,
)
from .graph import (
    _log_cycle_residuals,
    _solve_reference_energies_ls,
    _topological_transitions,
    _validate_transitions_graph,
)
from .hydrogen_variants import get_hydrogen_variants
from .model_compounds import MODEL_COMPOUNDS
from .models import TitratableResidueReference
from .types import HydrogenVariant, NamedTransition


# The model-compound assets live in ``opensqm/cph/model-compounds`` (one
# level up from this sub-package), so resolve relative to the *parent*
# directory rather than this module's folder.
_MODEL_COMPOUND_DIR = Path(__file__).resolve().parent.parent / "model-compounds"


def generate_ligand_reference(
    config: ConstantpHSettings,
    variant_molecules: list[Molecule],
    transitions: list[NamedTransition],
    ring_flip_bonds: list[tuple[str, str]] | None = None,
    iterations: int = 10,
    substeps: int = 20,
) -> TitratableResidueReference:
    """Compute (or load) a :class:`TitratableResidueReference` for a titratable ligand.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration shared with the production run; its
        ``hash()`` is part of the cache key.
    variant_molecules : list[openff.toolkit.topology.Molecule]
        Protonation states of the ligand with the *main variant* (the one
        the production topology will be built around) first. Each
        molecule's ``.name`` is used as its residue id in the topology
        (e.g. ``"LIG"``, ``"LIG1"``); names must be unique across variants.
        Heavy-atom names must be consistent across all states so the
        ``(h_name, parent_name)`` variant lists line up.
    transitions : list[NamedTransition]
        Microscopic 1-proton transitions ``(parent_name, child_name, pka)``,
        where ``parent_name`` and ``child_name`` are ``Molecule.name`` values
        from ``variant_molecules``. Pass an empty list when there is only one
        protomer (ring-flip MC may still be enabled via ``ring_flip_bonds``).
        Otherwise the transitions must span all variants from
        ``variant_molecules[0]``: at least ``len(variant_molecules) - 1``
        entries, every variant reachable from the root via directed
        parent->child arrows, and each edge dropping the formal charge by
        exactly 1.
    ring_flip_bonds : list[tuple[str, str]], optional
        Heavy-atom bonds whose 180-degree rotation should be proposed as
        ``ConstantPH`` terminal-group MC moves on this ligand. Each entry
        is a ``(anchor_atom_name, pivot_atom_name)`` pair using the
        topology atom names assigned by
        :func:`opensqm.cph.reference_energy.protonation_states._assign_ligand_atom_names`
        (i.e. ``N1``/``C5``/...). Typically obtained from
        :func:`opensqm.torsion_scanner.autodetect_flip_dihedrals_named`.
        This is *metadata* attached to the resulting
        :class:`TitratableResidueReference` -- it does not change the
        reference-energy fit -- so when a cached reference is loaded it
        is overridden by the user-supplied value (and the cache is
        rewritten if they differ). ``None`` is treated as an empty list.
    iterations, substeps : int
        Forwarded to :class:`ReferenceEnergyFinder`.

    Returns
    -------
    TitratableResidueReference
        With ``residue_name == main_variant == variant_molecules[0].name``
        and per-state ``(h_name, parent_name)`` lists in ``variants``,
        directly usable as a value in ``ConstantPH``'s ``residueVariants``.
    """
    if len(variant_molecules) < 1:
        raise ValueError("Need at least one variant")
    if not all(m.name for m in variant_molecules):
        raise ValueError("Every variant_molecule must have a non-empty .name")
    names = [m.name for m in variant_molecules]
    if len(set(names)) != len(names):
        raise ValueError(f"variant_molecule names must be unique, got {names}")

    transitions_resolved = _resolve_named_transitions(list(transitions), names)
    _validate_transitions_graph(
        transitions_resolved, len(variant_molecules), root_idx=0,
    )

    main_mol = variant_molecules[0]
    main_name = main_mol.name

    rdkit_mols = [m.to_rdkit() for m in variant_molecules]
    inchikeys = [to_inchikey_non_standard(rd) for rd in rdkit_mols]
    charges = [int(Chem.GetFormalCharge(rd)) for rd in rdkit_mols]
    inchikey_str = "|".join(inchikeys)

    xxhash_str = xxhash.xxh64(inchikey_str).hexdigest()
    # inchikey_str
    # Cache key encodes the full tree shape (parent/child indices + pKa) so
    # that two different ladders with the same set of pKas don't collide.
    edge_str = "-".join(
        f"{t.parent}>{t.child}@{np.round(t.pka, 1)}".replace(".", "")
        for t in transitions_resolved
    )

    reference_energies_dir = _MODEL_COMPOUND_DIR / "reference_energies"
    reference_energies_dir.mkdir(exist_ok=True)
    cache_path = (
        reference_energies_dir
        / f"{main_name}_{xxhash_str}_{edge_str}_{config.hash()}.json"
    )

    print(cache_path)

    user_ring_flip_bonds = [tuple(b) for b in (ring_flip_bonds or [])]
    if cache_path.exists():
        logger.info(f"Skipping {main_name} ({xxhash_str}): cached at {cache_path}")
        cached = TitratableResidueReference.load(cache_path)
        # ``ring_flip_bonds`` is cheap metadata, not part of the
        # reference-energy fit, so honour the user's input even when
        # loading a cached reference (and rewrite the cache so disk
        # reflects current intent for subsequent runs).
        if list(cached.ring_flip_bonds) != user_ring_flip_bonds:
            cached = cached.copy(update={"ring_flip_bonds": user_ring_flip_bonds})
            cached.save(cache_path)
        return cached

    per_state_variants: list[HydrogenVariant] = []
    for mol in variant_molecules:
        per_state_top = mol.to_topology().to_openmm()
        per_state_variants.append(get_hydrogen_variants(per_state_top)[0])

    lig_explicit_ff = config.get_explicit_forcefield(variant_molecules)

    omm_top, omm_pos = solvate_ligand(
        rdkit_mols[0], lig_explicit_ff, residue_name=main_name,
    )
    ligand_residues = [r for r in omm_top.residues() if r.name == main_name]
    if len(ligand_residues) != 1:
        raise ValueError(
            f"Solvated topology must contain exactly one ligand residue named "
            f"{main_name!r}; got {len(ligand_residues)}"
        )
    ligand_res_idx = ligand_residues[0].index

    # Iterate edges in BFS-from-root order so the per-pair sims log in a
    # readable parent-first order; for LS the order itself doesn't matter,
    # but keeping ``measured_deltas_kj`` aligned with ``ordered_transitions``
    # makes the residuals report easy to read alongside the same edges.
    ordered_transitions = _topological_transitions(transitions_resolved, root_idx=0)
    measured_deltas_kj: list[float] = []
    for t in ordered_transitions:
        pair_variants = [per_state_variants[t.parent], per_state_variants[t.child]]
        pair_names = [names[t.parent], names[t.child]]
        pair_charges = [charges[t.parent], charges[t.child]]
        pair_label = f"{names[t.parent]}-{names[t.child]}"
        pair_reference = _make_pair_reference(
            residue_name=main_name,
            pair_variants=pair_variants,
            pair_names=pair_names,
            pair_charges=pair_charges,
            pka=t.pka,
        )
        cph = ConstantPH(
            topology=omm_top,
            positions=omm_pos,
            pH=7.0,
            config=config,
            references={main_name: pair_reference},
            titratable_residue_indices=[ligand_res_idx],
            ligand_variant_molecules=variant_molecules,
            ring_flip_angles=None,
        )
        logger.info(f"Computing reference energy for {pair_label} (pKa={t.pka})")
        finder = ReferenceEnergyFinder(cph, pKa=t.pka, temperature=config.temperature)
        finder.findReferenceEnergies(iterations=iterations, substeps=substeps)
        ref_energies = cph.titrations[ligand_res_idx].referenceEnergies
        logger.info(
            f"Computed reference energy for {pair_label} (pKa={t.pka}): {ref_energies}"
        )
        measured_deltas_kj.append(float(ref_energies[1]._value))

    energies_kj, edge_residuals_kj = _solve_reference_energies_ls(
        ordered_transitions,
        measured_deltas_kj,
        n_variants=len(variant_molecules),
        root_idx=0,
    )
    _log_cycle_residuals(main_name, ordered_transitions, edge_residuals_kj, names)

    reference = TitratableResidueReference(
        residue_name=main_name,
        main_variant=main_name,
        variant_names=list(names),
        variants=per_state_variants,
        charges=charges,
        reference_energies_kj_per_mole=energies_kj,
        transitions=transitions_resolved,
        ring_flip_bonds=user_ring_flip_bonds,
    )
    reference.save(cache_path)
    logger.info(f"Saved {main_name} -> {cache_path}")
    return reference


def generate_residue_reference_dict(
    config: ConstantpHSettings,
    ligands: list[
        tuple[list[Molecule], list[NamedTransition]]
        | tuple[list[Molecule], list[NamedTransition], list[tuple[str, str]]]
    ] | None = None,
    iterations: int = 20_000,
    substeps: int = 20,
) -> dict[str, TitratableResidueReference]:
    """Generate and cache :class:`TitratableResidueReference` objects for all model compounds.

    For each entry in :data:`MODEL_COMPOUNDS`, walk the transitions tree
    rooted at ``main_variant`` and compute (or load from disk) the
    reference energy of every non-main variant relative to it. The
    assembled object is written to disk as JSON; subsequent calls with
    the same ``config`` hash will load that JSON directly without
    re-running the per-edge Monte Carlo finder.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration. Forcefield definitions, temperature, and
        the ligand parameterisation choices (``partial_charge_method``,
        ``bespoke_ligand_forcefield``) all participate in the cache hash.
    ligands : list, optional
        Each entry is a 2-tuple ``(variant_molecules, transitions)`` or a
        3-tuple ``(variant_molecules, transitions, ring_flip_bonds)``
        forwarded to :func:`generate_ligand_reference`. ``transitions``
        is a list of ``(parent_name, child_name, pka)`` named-tuple
        entries; see that function's docstring for the spanning-tree
        contract. ``ring_flip_bonds`` is an optional list of
        ``(anchor_atom_name, pivot_atom_name)`` pairs (typically the
        output of :func:`opensqm.torsion_scanner.autodetect_flip_dihedrals_named`)
        recording rotatable bonds that ``ConstantPH`` should treat as
        terminal-group MC moves.
    iterations : int
        Number of Monte Carlo moves to attempt per pathway.
    substeps : int
        Number of dynamics steps to integrate between Monte Carlo moves.

    Returns
    -------
    dict[str, TitratableResidueReference]
        Keyed by residue name (e.g. ``"HIS"`` for proteins, the ligand's
        ``Molecule.name`` for ligands).
    """
    reference_energies_dir = _MODEL_COMPOUND_DIR / "reference_energies"
    reference_energies_dir.mkdir(exist_ok=True)

    references: dict[str, TitratableResidueReference] = {}

    for residue_name, info in MODEL_COMPOUNDS.items():
        pdb_name = info["pdb_name"]
        main_variant = info["main_variant"]
        variants = list(info["variants"])
        charges = list(info["charges"])
        raw_transitions = list(info["transitions"])
        ring_flip_bonds = [tuple(b) for b in info.get("ring_flip_bonds", [])]

        if variants[0] != main_variant:
            raise ValueError(
                f"{residue_name}: variants[0] ({variants[0]!r}) must equal "
                f"main_variant ({main_variant!r})"
            )
        if len(charges) != len(variants):
            raise ValueError(
                f"{residue_name}: charges length ({len(charges)}) must equal "
                f"variants length ({len(variants)})"
            )
        transitions_resolved = _resolve_named_transitions(raw_transitions, variants)
        _validate_transitions_graph(transitions_resolved, len(variants), root_idx=0)

        reference_path = (
            reference_energies_dir / f"{residue_name}_{config.hash()}.json"
        )
        if reference_path.exists():
            logger.info(f"Skipping {residue_name}: cached at {reference_path}")
            cached = TitratableResidueReference.load(reference_path)
            # As for ligands, ``ring_flip_bonds`` is cheap metadata that
            # doesn't participate in the reference-energy fit, so honour
            # whatever ``MODEL_COMPOUNDS`` currently declares even when
            # loading legacy caches written before the field existed.
            if list(cached.ring_flip_bonds) != ring_flip_bonds:
                cached = cached.copy(update={"ring_flip_bonds": ring_flip_bonds})
                cached.save(reference_path)
            references[residue_name] = cached
            continue

        ordered_transitions = _topological_transitions(transitions_resolved, root_idx=0)
        measured_deltas_kj: list[float] = []
        for t in ordered_transitions:
            measured_deltas_kj.append(
                _compute_pairwise_reference_energy(
                    model_compound_dir=_MODEL_COMPOUND_DIR,
                    pdb_name=pdb_name,
                    residue_name=residue_name,
                    pair=[variants[t.parent], variants[t.child]],
                    pair_charges=[charges[t.parent], charges[t.child]],
                    pka=t.pka,
                    config=config,
                    iterations=iterations,
                    substeps=substeps,
                )
            )
        energies_kj, edge_residuals_kj = _solve_reference_energies_ls(
            ordered_transitions,
            measured_deltas_kj,
            n_variants=len(variants),
            root_idx=0,
        )
        _log_cycle_residuals(residue_name, ordered_transitions, edge_residuals_kj, variants)

        reference = TitratableResidueReference(
            residue_name=residue_name,
            main_variant=main_variant,
            variant_names=list(variants),
            variants=variants,
            charges=charges,
            reference_energies_kj_per_mole=energies_kj,
            transitions=transitions_resolved,
            ring_flip_bonds=ring_flip_bonds,
        )
        reference.save(reference_path)
        logger.info(f"Saved {residue_name} -> {reference_path}")
        references[residue_name] = reference

    if ligands is not None:

        for entry in ligands:
            if len(entry) == 2:
                variant_molecules, transitions = entry
                lig_ring_flip_bonds: list[tuple[str, str]] = []
            elif len(entry) == 3:
                variant_molecules, transitions, lig_ring_flip_bonds = entry
            else:
                raise ValueError(
                    f"each ligand entry must be a 2-tuple "
                    f"(variant_molecules, transitions) or a 3-tuple "
                    f"(variant_molecules, transitions, ring_flip_bonds); "
                    f"got an entry of length {len(entry)}"
                )

            name = variant_molecules[0].name
            reference = generate_ligand_reference(
                config=config,
                variant_molecules=variant_molecules,
                transitions=transitions,
                ring_flip_bonds=list(lig_ring_flip_bonds),
                iterations=iterations,
                substeps=substeps,
            )
            references[name] = reference

    return references


__all__ = ["generate_ligand_reference", "generate_residue_reference_dict"]
