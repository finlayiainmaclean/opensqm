"""Top-level driver for constant-pH REMD runs: setup, production, and analysis."""

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mdtraj as md
import numpy as np
import pandas as pd
from loguru import logger
from openff.toolkit.topology import Molecule  # type: ignore
from openmm import Context, LangevinIntegrator, MonteCarloBarostat, System, unit
from openmm.app import CutoffNonPeriodic, ForceField, HBonds, Modeller, PDBFile, Topology
from pydantic import BaseModel, ConfigDict, Field
from pydantic_units import OpenMMQuantity
from rdkit import Chem
from tqdm import tqdm
from unipka import UnipKa

from opensqm.cph.checkpoint import (
    PRODUCTION_STATE_FILENAME,
    REMD_STATE_FILENAME,
    checkpoint_dir,
    equilibrated_pdb_path,
    read_production_state,
    read_run_manifest,
    update_manifest_weights,
    validate_run_manifest,
    write_production_state,
    write_run_manifest,
)
from opensqm.cph.constantph import (
    residue_label,
    select_titratable_residue_indices,
    write_solute_pdb,
)
from opensqm.cph.mmgbsa import _batch_index_for_step, _snap_ph, compute_replica_mmgbsa
from opensqm.cph.ph_remd import ConstantPHRemd
from opensqm.cph.pka import (
    analyze_cph_results,
    build_replica_overlay_timeseries,
    compute_pka_timeseries,
)
from opensqm.cph.propka import find_titratable_residues_near_ph, predict_pkas
from opensqm.cph.reference_energy import (
    build_protonation_states,
    build_transitions_tree,
    generate_residue_reference_dict,
)
from opensqm.cph.simulation_config import ConstantpHSettings
from opensqm.cph.trajectory import (
    StateSplitTrajectoryManager,
    iter_replica_state_trajectories,
    replica_trajectory_base,
    replica_trajectory_index,
    trajectories_dir,
)
from opensqm.md.charge_cache import load_ligand_variant, persist_ligand_setups
from opensqm.md.equilibrate import EquilibrationSettings, equilibrate
from opensqm.md.prepare import prepare_protein, prepare_system
from opensqm.rdkit_utils import set_residue_info
from opensqm.torsion_scanner import autodetect_flip_dihedrals_named

if TYPE_CHECKING:
    from opensqm.cph.constantph import ConstantPH


def _default_equilibration_config() -> EquilibrationSettings:
    return EquilibrationSettings(
        npt_time=100 * unit.picoseconds,
        warmup_time=10 * unit.picoseconds,
    )


class ConstantpHRunSettings(BaseModel):
    """Settings for a constant-pH MD run."""

    model_config = ConfigDict(frozen=True)
    # Equilibration settings
    equilibration_config: EquilibrationSettings = Field(
        default_factory=_default_equilibration_config
    )
    # Production settings
    production_time: OpenMMQuantity[unit.nanosecond] = 10 * unit.nanosecond
    integrator_step_size: OpenMMQuantity[unit.picosecond] = 0.004 * unit.picoseconds
    barostat_pressure: OpenMMQuantity[unit.bar] | None = None
    reporter_interval: OpenMMQuantity[unit.picosecond] = 1 * unit.picosecond
    # CPH settings
    cph_config: ConstantpHSettings = Field(default_factory=ConstantpHSettings)
    titratable_residue_query: str | None = None
    # Intersect the query selection with the residues PROPKA predicts are near
    # their pKa at some pH in the ladder, so only residues that are both
    # requested and actually titrating are driven. The ligand/cofactor are
    # exempt (PROPKA only sees the protein). See ``_propka_titratable_indices``.
    restrict_titratable_to_near_ph: bool = True

    ligand_terminal_ring_mc: bool = False
    mmgbsa_n_closest_waters: int = 5
    ligand_protonation: bool = True
    # Seed each titratable residue's starting protonation near equilibrium
    # instead of fully protonated: PROPKA-predicted protonation for protein
    # residues and the dominant uniKa protomer for the ligand/cofactor. Only
    # sets the initial state (the MC still samples freely from there), so it
    # does not change converged results - just avoids a long titrate-down
    # burn-in and stops short/poorly-mixed runs reporting the fully-protonated
    # start. See ``_seed_initial_variant_indices``.
    seed_initial_protonation: bool = True
    protonation_penalty: OpenMMQuantity[unit.kilocalories_per_mole] = (
        2.0 * unit.kilocalories_per_mole
    )
    protonation_swap_interval: OpenMMQuantity[unit.picosecond] = 0.2 * unit.picoseconds
    remd_swap_interval: OpenMMQuantity[unit.picosecond] = 10 * unit.picoseconds
    use_ph_remd: bool = True
    n_replicas: int = 3
    ph: float | list[float] = Field(default_factory=lambda: [1.0, 7.0, 14.0])
    # pH at which ligand/cofactor protomers are ENUMERATED (the uniKa
    # distribution used to pick which protonation states to model). Defaults to
    # ``ph`` - so a multi-pH pKa ladder enumerates every protomer relevant
    # across the ladder. Set this to a single pH to decouple enumeration from
    # sampling: e.g. a tight REMD ladder around pH 7 for better sampling, while
    # still enumerating only the pH-7-relevant protomers (avoids a protomer
    # explosion and the associated reference-energy cost).
    protomer_enumeration_ph: float | None = None


@dataclass(frozen=True)
class LigandSetup:
    """Protonation-variant molecules, transitions, and ring-flip bonds for a small molecule.

    ``union_molecule`` is the maximally-protonated super-template (every
    titration site protonated at once). It is *not* a sampled protonation
    state; it exists only so its SMIRNOFF template can be registered on the
    constant-pH force field, letting ``ConstantPH`` build the master topology
    from the union of all variants' titratable hydrogens (needed when the
    in-window protomers are non-nested siblings). ``None`` when the union
    coincides with a real variant (nested ladder) or when protonation
    enumeration is disabled - in those cases no extra template is needed.

    ``dominant_variant_index`` is the index (into ``variant_molecules``) of the
    most populated protomer in the uniKa distribution at the run pH, used to
    seed the ligand's starting protonation state. ``None`` when unknown (e.g. a
    resumed setup rebuilt from the variant CSV).
    """

    variant_molecules: list[Molecule]
    transitions: list[Any]
    ring_flip_bonds: list[tuple[str, str]]
    union_molecule: Molecule | None = None
    dominant_variant_index: int | None = None


def _build_small_molecule_variants(
    molecule_path: str,
    config: ConstantpHRunSettings,
    output_path: Path,
    unipka: UnipKa,
    *,
    residue_name: str,
    csv_name: str,
) -> LigandSetup:
    molecule_rdmol = set_residue_info(Chem.MolFromMolFile(molecule_path, removeHs=False))

    if config.ligand_protonation:
        # Enumerate protomers at a single pH when requested, else across the
        # (possibly multi-pH) sampling ladder. Decoupling lets a tight REMD
        # ladder improve sampling without pulling in ladder-edge protomers.
        enumeration_ph = (
            config.protomer_enumeration_ph
            if config.protomer_enumeration_ph is not None
            else config.ph
        )
        molecule_distribution = unipka.get_distribution(
            molecule_rdmol, pH=enumeration_ph
        ).reset_index(drop=True)
        ligand_protonation_penalty = config.protonation_penalty.value_in_unit(
            unit.kilocalories_per_mole
        )
        molecule_distribution = molecule_distribution[
            molecule_distribution["relative_ph_adjusted_free_energy"] < ligand_protonation_penalty
        ]
        molecule_distribution = molecule_distribution.sort_values(
            ["charge", "relative_ph_adjusted_free_energy"],
            ascending=[False, True],
        ).reset_index(drop=True)
        molecule_mols, template_rdmol = build_protonation_states(
            list(molecule_distribution["mol"]),
            geometry_mol=molecule_rdmol,
            return_template=True,
        )
        logger.info(f"Number of {residue_name} protonation states: {len(molecule_mols)}")
        print(molecule_distribution[['smiles','population']].T)
        # Most populated protomer at the run pH (variants share the sorted,
        # reset-index order of molecule_distribution) - used to seed the
        # ligand's starting protonation state in the constant-pH run.
        dominant_variant_index = int(np.asarray(molecule_distribution["population"]).argmax())
    else:
        molecule_mols = [molecule_rdmol]
        template_rdmol = None
        dominant_variant_index = 0

    variant_molecules: list[Molecule] = []
    for i, rdmol in enumerate(molecule_mols):
        offmol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
        offmol.name = residue_name if i == 0 else f"{residue_name}{i}"
        variant_molecules.append(offmol)

    # The super-template protonates every titration site at once. When the
    # in-window protomers are non-nested siblings (each protonating a
    # different site), no single variant is a superset, so ConstantPH needs
    # this union protomer registered on the force field to parametrise the
    # master topology. Skip it when the union coincides with a real variant
    # (nested ladder): registering an isomorphic duplicate would make SMIRNOFF
    # template matching ambiguous.
    union_molecule: Molecule | None = None
    if template_rdmol is not None:
        candidate = Molecule.from_rdkit(template_rdmol, allow_undefined_stereo=True)
        candidate.name = f"{residue_name}_UNION"
        if not any(candidate.is_isomorphic_with(v) for v in variant_molecules):
            union_molecule = candidate

    pd.DataFrame(
        {
            "residue_name": [mol.name for mol in variant_molecules],
            "smiles": [Chem.MolToSmiles(mol) for mol in molecule_mols],
        }
    ).to_csv(output_path / csv_name, index=False)

    if config.ligand_terminal_ring_mc:
        ring_flip_bonds = autodetect_flip_dihedrals_named(molecule_mols[0])
        logger.info(f"{residue_name} ring_flip_bonds: {ring_flip_bonds}")
    else:
        ring_flip_bonds = []

    transitions = build_transitions_tree(
        variant_molecules,
        pka_fn=lambda p, c: unipka.get_macro_pka_from_macrostates(
            acid_macrostate=[p],
            base_macrostate=[c],
        ),
    )
    return LigandSetup(
        variant_molecules,
        transitions,
        ring_flip_bonds,
        union_molecule,
        dominant_variant_index,
    )


def _load_or_build_small_molecule_variants(
    molecule_path: str | None,
    csv_path: Path,
    output_path: Path,
    config: ConstantpHRunSettings,
    unipka: UnipKa,
    *,
    residue_name: str,
) -> LigandSetup | None:
    """Reload variant setup from CSV, or rebuild from the source file if missing."""
    if molecule_path is None:
        return None
    setup = _reload_ligand_setup_from_csv(csv_path, output_path, config, unipka)
    if setup is not None:
        return setup
    logger.info(f"{csv_path.name} not found; rebuilding {residue_name} variants from file")
    return _build_small_molecule_variants(
        molecule_path,
        config,
        output_path,
        unipka,
        residue_name=residue_name,
        csv_name=csv_path.name,
    )


def _reload_ligand_setup_from_csv(
    csv_path: Path,
    output_path: Path,
    config: ConstantpHRunSettings,
    unipka: UnipKa,
) -> LigandSetup | None:
    """Rebuild ligand/cofactor setup from a prior run's variant CSV."""
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    variant_molecules: list[Molecule] = []
    molecule_mols = []
    for name, smiles in zip(df["residue_name"], df["smiles"], strict=True):
        residue_name = str(name)
        cached = load_ligand_variant(output_path, residue_name)
        if cached is not None and cached.partial_charges is not None:
            cached.name = residue_name
            variant_molecules.append(cached)
            molecule_mols.append(Chem.MolFromSmiles(str(smiles)))
            continue

        rdmol = Chem.MolFromSmiles(str(smiles))
        molecule_mols.append(rdmol)
        offmol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
        offmol.name = residue_name
        variant_molecules.append(offmol)

    if config.ligand_terminal_ring_mc:
        ring_flip_bonds = autodetect_flip_dihedrals_named(molecule_mols[0])
    else:
        ring_flip_bonds = []

    transitions = build_transitions_tree(
        variant_molecules,
        pka_fn=lambda p, c: unipka.get_macro_pka_from_macrostates(
            acid_macrostate=[p],
            base_macrostate=[c],
        ),
    )

    # Reload the persisted union super-template (named "<MAIN>_UNION"), if the
    # original run created one; it is only present for non-nested ladders.
    main_name = str(df["residue_name"].iloc[0])
    union_molecule = load_ligand_variant(output_path, f"{main_name}_UNION")
    if union_molecule is not None:
        union_molecule.name = f"{main_name}_UNION"

    return LigandSetup(variant_molecules, transitions, ring_flip_bonds, union_molecule)


def _ligand_entries_from_setups(
    ligand_setup: LigandSetup | None,
    cofactor_setup: LigandSetup | None,
) -> list[tuple[list[Molecule], list[Any], list[tuple[str, str]]]]:
    entries = []
    for setup in (ligand_setup, cofactor_setup):
        if setup is None:
            continue
        entries.append((setup.variant_molecules, setup.transitions, setup.ring_flip_bonds))
    return entries


def _log_titratable_residues(topology: Topology, titratable_residue_indices: list[int]) -> None:
    residues_by_index = {r.index: r for r in topology.residues()}
    formatted = ", ".join(
        residue_label(residues_by_index[i]) for i in titratable_residue_indices
    )
    logger.info(f"Titratable residues: {formatted}")


def _variant_index_from_pka(reference: Any, pka: float, ph: float) -> int:
    """Pick the protonation variant matching a PROPKA pKa at ``ph``.

    Below the pKa the group is majority protonated, so choose the
    highest-charge (most-protonated) variant; at or above it, majority
    deprotonated, so choose the lowest-charge one. For a two-neutral-tautomer
    case (e.g. neutral HIS), ``argmin`` deterministically picks the first.
    """
    charges = list(reference.charges)
    return int(np.argmax(charges)) if ph < pka else int(np.argmin(charges))


def _seed_initial_variant_indices(
    topology: Topology,
    residue_reference_dict: dict,
    titratable_residue_indices: list[int],
    protein_pdb: str | Path,
    ph: float,
    ligand_setup: LigandSetup | None,
    cofactor_setup: LigandSetup | None,
) -> dict[int, int]:
    """Map each titratable residue to a near-equilibrium starting variant index.

    Protein residues are seeded from PROPKA's predicted protonation at ``ph``
    (matched to the topology by chain + residue number); the ligand/cofactor
    are seeded to their dominant uniKa protomer. Residues that cannot be
    resolved are omitted, so ``ConstantPH`` leaves them at its fully-protonated
    default.
    """
    titratable = set(titratable_residue_indices)
    seeds: dict[int, int] = {}

    # Ligand / cofactor: dominant uniKa protomer.
    for setup, resname in ((ligand_setup, "LIG"), (cofactor_setup, "COF")):
        if setup is None or setup.dominant_variant_index is None:
            continue
        for res in topology.residues():
            if res.index in titratable and res.name == resname:
                seeds[res.index] = setup.dominant_variant_index

    # Protein: PROPKA-predicted protonation, matched by (chain, residue number).
    try:
        predictions = predict_pkas(protein_pdb, ph=ph)
    except Exception as exc:  # PROPKA is best-effort; keep the ligand seeds
        logger.warning(f"PROPKA seeding failed ({exc}); protein residues start fully protonated")
        return seeds
    pred_by_key = {(p.chain_id, p.residue_number): p for p in predictions}
    for res in topology.residues():
        if res.index not in titratable or res.index in seeds:
            continue
        reference = residue_reference_dict.get(res.name)
        if reference is None:
            continue
        chain_id = (res.chain.id or "").strip()
        try:
            resnum = int(res.id)
        except (TypeError, ValueError):
            continue
        prediction = pred_by_key.get((chain_id, resnum)) or pred_by_key.get(("", resnum))
        if prediction is None:
            continue
        seeds[res.index] = _variant_index_from_pka(reference, prediction.pka, ph)
    return seeds


def _target_ph_variant_probabilities(
    reference: Any, ph: float, temperature: unit.Quantity
) -> np.ndarray:
    """Boltzmann probability over a residue's variants at ``ph`` (isolated).

    ``p(i) ∝ exp(reference_energy[i]/kT - charge[i]·ln(10)·pH)`` - the stationary
    distribution of the MC acceptance with the environment term dropped, using
    formal charge as the proton-count proxy (Δprotons == Δcharge for a
    protonation change). This is the "energy penalty from the target pH" weight:
    the target-pH-favoured state gets the most weight, strongly-penalised
    protomers little.
    """
    kT = (unit.MOLAR_GAS_CONSTANT_R * temperature).value_in_unit(unit.kilojoules_per_mole)
    energies = np.asarray(reference.reference_energies_kj_per_mole, dtype=float)
    charges = np.asarray(reference.charges, dtype=float)
    logits = energies / kT - charges * np.log(10.0) * ph
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def _replica_initial_variant_indices(
    deterministic_seed: dict[int, int],
    topology: Topology,
    residue_reference_dict: dict,
    titratable_residue_indices: list[int],
    ph: float,
    temperature: unit.Quantity,
    n_replicas: int,
) -> list[dict[int, int]]:
    """One starting-variant map per replica for diversified initial conditions.

    Replica 0 keeps ``deterministic_seed`` (the PROPKA/dominant best guess);
    replicas 1+ draw each residue's starting variant from its target-pH
    Boltzmann distribution (:func:`_target_ph_variant_probabilities`). This
    spreads the initial replica ensemble across macrostates weighted by energy -
    likely states often, high-penalty ones rarely - so pH-REMD has real
    macrostate diversity to exchange instead of every replica starting
    identically (which low-acceptance MC alone would never diversify).
    """
    seeds = [dict(deterministic_seed)]
    if n_replicas <= 1:
        return seeds
    residues = list(topology.residues())
    probs: dict[int, np.ndarray] = {}
    for res_index in titratable_residue_indices:
        reference = residue_reference_dict.get(residues[res_index].name)
        if reference is not None and len(reference.charges) > 1:
            probs[res_index] = _target_ph_variant_probabilities(reference, ph, temperature)
    for _ in range(1, n_replicas):
        seeds.append(
            {
                res_index: int(np.random.choice(len(p), p=p))
                for res_index, p in probs.items()
            }
        )
    return seeds


def _propka_titratable_indices(
    protein_pdb: str | Path,
    topology: Topology,
    allowed_indices: set[int],
    phs: list[float],
    *,
    protonation_penalty: unit.Quantity,
    temperature: unit.Quantity,
) -> set[int]:
    """Topology residue indices PROPKA flags as near-titrating at any pH.

    For each pH in ``pHs``, PROPKA is run on the dry protein ``protein_pdb`` and
    every group whose minor protonation state lies within ``protonation_penalty``
    of that pH is matched back to its topology residue. Only indices in
    ``allowed_indices`` (residues that have a titratable reference and are not
    disulfide-bonded) can be returned, so PROPKA never introduces a residue the
    constant-pH model has no states for.

    Residues are matched on ``(chain, resSeq, resname)``; when chain ids differ
    between PROPKA and the topology (e.g. blank chains) a unique
    ``(resSeq, resname)`` match is used as a fallback.
    """
    by_full_key: dict[tuple[str, int, str], int] = {}
    by_num_name: dict[tuple[int, str], list[int]] = defaultdict(list)
    for residue in topology.residues():
        if residue.index not in allowed_indices:
            continue
        try:
            res_num = int(residue.id)
        except (TypeError, ValueError):
            continue
        chain_id = (residue.chain.id or "").strip()
        by_full_key[(chain_id, res_num, residue.name)] = residue.index
        by_num_name[(res_num, residue.name)].append(residue.index)

    indices: set[int] = set()
    for ph in phs:
        for prediction in find_titratable_residues_near_ph(
            protein_pdb,
            ph=ph,
            protonation_penalty=protonation_penalty,
            temperature=temperature,
        ):
            key = (prediction.chain_id, prediction.residue_number, prediction.residue_name)
            index = by_full_key.get(key)
            if index is None:
                candidates = by_num_name.get(
                    (prediction.residue_number, prediction.residue_name), []
                )
                if len(candidates) == 1:
                    index = candidates[0]
            if index is not None:
                indices.add(index)
    return indices


def _apply_barostat(remd: ConstantPHRemd, config: ConstantpHRunSettings) -> None:
    if config.barostat_pressure is None:
        return
    for replica in remd.replicas:
        replica.simulation.system.addForce(
            MonteCarloBarostat(config.barostat_pressure, 300 * unit.kelvin)
        )
        # reinitialize rebuilds the context from the System, reverting the
        # per-particle protonation parameters to the System's maximal state, so
        # re-apply the current protonation variants afterwards.
        replica.simulation.context.reinitialize(preserveState=True)
        replica.apply_current_states()


def _optimize_remd_weights(remd: ConstantPHRemd) -> np.ndarray:
    cur_weights = np.array(remd.weights)
    logger.info("Optimising weights")
    num_successful_swaps = 0
    equilibration_mc_steps = 2000
    with tqdm(range(equilibration_mc_steps), desc="Equilibration") as pbar:
        for _ in pbar:
            prev_weights = np.array(cur_weights)
            remd.step(50)  # 0.2ps
            swaps = remd.attempt_mc_step()
            num_successful_swaps += sum(int(swap) for swap in swaps)
            cur_weights = np.array(remd.weights)
            diff = np.linalg.norm(cur_weights - prev_weights)
            pbar.set_postfix(diff=f"{diff:.2f}")

            if diff < 1e-1:
                logger.info("Converged")
                break
        else:
            logger.warning("Did not converge")
    logger.info(f"Number of successful swaps: {num_successful_swaps}")
    logger.info(f"Current weights: {cur_weights}")
    return cur_weights


@dataclass(kw_only=True)
class SystemState:
    """A simulation state: topology, positions, and optionally a parametrized System.

    Produced by CpH as a minimum-energy frame (``system`` left ``None`` since the
    stripped per-state sub-topology has no matching system - the consumer builds
    one), and reused as the base of modbinddg's ``PreparedState``. ``ligand``
    optionally carries the LIG residue as an RDKit mol in the exact protonation
    state of this frame, so the system can be rebuilt with the right protomer.
    """

    topology: Topology  # box vectors set to this frame's cell
    positions: unit.Quantity  # (n_atoms, 3), physical atoms of that protonation state
    system: System | None = None
    ligand: "Chem.Mol | None" = None


@dataclass
class PHResult:
    """Per-pH summary for downstream use (e.g. modbinddg/mmgbsa)."""

    ph: float
    population: float  # joint population fraction of the snapshot's state at this pH
    lowest_energy_snapshot: SystemState | None


def _per_frame_system_energies(
    output_path: Path,
    replica_dfs: list[pd.DataFrame],
    cph_config: ConstantpHSettings,
    ph_ladder: list[float],
    protonation_swap_steps: int,
) -> list[dict]:
    """Compute implicit-solvent potential energy per frame for each state DCD.

    Used when no ligand is present. Builds a fresh OpenMM context from each
    per-state PDB topology and evaluates the GBn2 potential energy for every
    frame, grouped by the pH active at that batch step.
    """
    records: list[dict] = []
    for replica_i, replica_df in enumerate(replica_dfs):
        # Force ``system_state`` to string: with a single titratable residue the
        # column holds only single integers ("0"/"1"), which pandas would infer
        # as int64, so the ``== state_label`` (a string from the DCD filename)
        # comparison below would match nothing and silently yield zero frames.
        traj_csv = pd.read_csv(
            replica_trajectory_index(output_path, replica_i),
            dtype={"system_state": str},
        )
        batch_ph = replica_df["ph"]

        for dcd_path, pdb_path in iter_replica_state_trajectories(output_path, replica_i):
            state_label = dcd_path.name.split(".", 1)[-1].removesuffix(".dcd")
            state_rows = traj_csv[traj_csv["system_state"] == state_label]
            if state_rows.empty:
                continue

            pdb = PDBFile(str(pdb_path))
            ff = ForceField(*cph_config.implicit_ff_files)
            system = ff.createSystem(
                pdb.topology, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds
            )
            integrator = LangevinIntegrator(
                cph_config.temperature, cph_config.friction, cph_config.timestep
            )
            context = Context(system, integrator)

            traj = md.load_dcd(str(dcd_path), top=str(pdb_path))
            frame_to_time = dict(
                zip(state_rows["frame_ix"].astype(int), state_rows["time_ns"], strict=False)
            )
            frame_to_step = dict(
                zip(
                    state_rows["frame_ix"].astype(int), state_rows["step"].astype(int), strict=False
                )
            )

            for frame_ix in range(traj.n_frames):
                if frame_ix not in frame_to_time:
                    continue
                step = frame_to_step[frame_ix]
                batch_idx = _batch_index_for_step(step, protonation_swap_steps, len(batch_ph))
                ph = _snap_ph(float(batch_ph.iloc[batch_idx]), ph_ladder)
                context.setPositions(traj[frame_ix].xyz[0] * unit.nanometers)
                energy = (
                    context.getState(getEnergy=True)
                    .getPotentialEnergy()
                    .value_in_unit(unit.kilocalories_per_mole)
                )
                records.append(
                    {
                        "ph": ph,
                        "state_label": state_label,
                        "time_ns": frame_to_time[frame_ix],
                        "replica_i": replica_i,
                        "energy": energy,
                        "energy_type": "potential",
                        "dcd_path": dcd_path,
                        "pdb_path": pdb_path,
                        "frame_ix": frame_ix,
                    }
                )
    return records


def _ligand_rdmol_for_state(
    cph: "ConstantPH | None",
    ligand_variant_molecules: "list[Molecule] | None",
    state_label: str,
    ligand_resname: str = "LIG",
) -> "Chem.Mol | None":
    """Return the ligand RDKit mol in the protonation state encoded by ``state_label``.

    ``state_label`` is the trajectory manager's per-titration variant-index
    signature ("i_j_k"): the indices align with ``sorted(cph.titrations)``, and
    for the ligand residue the variant index maps 1:1 onto
    ``ligand_variant_molecules`` (see ``generate_ligand_reference``). Returns
    ``None`` if the ligand cannot be resolved unambiguously.
    """
    if not ligand_variant_molecules or cph is None:
        return None
    keys = sorted(cph.titrations)
    try:
        indices = [int(x) for x in str(state_label).split("_")]
    except ValueError:
        return None
    if len(indices) != len(keys):
        return None
    resname_by_index = {r.index: r.name for r in cph.explicitTopology.residues()}
    ligand_keys = [k for k in keys if str(resname_by_index.get(k, "")).startswith(ligand_resname)]
    if len(ligand_keys) != 1:
        return None
    variant_index = indices[keys.index(ligand_keys[0])]
    if not 0 <= variant_index < len(ligand_variant_molecules):
        return None
    return ligand_variant_molecules[variant_index].to_rdkit()


def _state_label_to_joint_label(cph: "ConstantPH | None", state_label: str) -> "str | None":
    """Convert a frame signature ("0_0_1_...") to the joint-population column label.

    Mirrors ``compute_joint_populations._joint_label`` (per-residue
    ``"<resname>.<idx>:<variant>"`` joined by " | ", ordered by
    ``sorted(cph.titrations)``) so a frame's microstate can be matched against
    the joint-population table, whose columns are these human-readable labels.
    Returns ``None`` if the label cannot be mapped.
    """
    if cph is None:
        return None
    titratable_indices = sorted(cph.titrations.keys())
    try:
        indices = [int(x) for x in str(state_label).split("_")]
    except ValueError:
        return None
    if len(indices) != len(titratable_indices):
        return None
    resname_by_index = {r.index: r.name for r in cph.explicitTopology.residues()}
    parts = []
    for res_idx, state in zip(titratable_indices, indices, strict=False):
        titration = cph.titrations[res_idx]
        if not 0 <= state < len(titration.variant_names):
            return None
        parts.append(f"{resname_by_index.get(res_idx)}.{res_idx}:{titration.variant_names[state]}")
    return " | ".join(parts)


def _build_ph_results(
    analysis: dict,
    frame_records: list[dict],
    ph_ladder: list[float],
    last_fraction: float = 0.1,
    *,
    cph: "ConstantPH | None" = None,
    ligand_variant_molecules: "list[Molecule] | None" = None,
) -> list[PHResult]:
    """Build per-pH summaries of the dominant microstate and a representative frame.

    For each pH, report the dominant microstate's population and a
    representative (lowest-energy) frame drawn from that microstate.
    """
    joint_pops_df: pd.DataFrame = analysis.get("joint_populations", pd.DataFrame())

    if frame_records:
        valid_times = [r["time_ns"] for r in frame_records if not np.isnan(r["time_ns"])]
        cutoff = max(valid_times) * (1.0 - last_fraction) if valid_times else float("inf")
        tail = [r for r in frame_records if r["time_ns"] >= cutoff]
    else:
        tail = []

    results: list[PHResult] = []
    for ph in ph_ladder:
        row = (
            joint_pops_df.loc[ph]
            if (not joint_pops_df.empty and ph in joint_pops_df.index)
            else None
        )
        # The dominant microstate at this pH is the highest-population column.
        dominant_label = str(row.idxmax()) if row is not None and not row.empty else None

        ph_tail = [r for r in tail if abs(r["ph"] - ph) < 1e-6]
        # Prefer frames sampled in the dominant microstate; fall back to all
        # frames at this pH if none were recorded for it.
        preferred = (
            [
                r
                for r in ph_tail
                if _state_label_to_joint_label(cph, r["state_label"]) == dominant_label
            ]
            if dominant_label is not None
            else []
        )
        candidates = preferred or ph_tail

        snapshot: SystemState | None = None
        population = 0.0
        if candidates:
            best = min(candidates, key=lambda r: r["energy"])
            frame = md.load_frame(
                str(best["dcd_path"]), best["frame_ix"], top=str(best["pdb_path"])
            )
            topology = PDBFile(str(best["pdb_path"])).topology
            if frame.unitcell_vectors is not None:
                topology.setPeriodicBoxVectors(frame.unitcell_vectors[0] * unit.nanometer)
            snapshot = SystemState(
                topology=topology,
                positions=frame.xyz[0] * unit.nanometer,
                ligand=_ligand_rdmol_for_state(cph, ligand_variant_molecules, best["state_label"]),
            )
            # Report the population of the microstate the chosen frame is in.
            chosen_label = _state_label_to_joint_label(cph, best["state_label"])
            if row is not None and chosen_label is not None and chosen_label in row.index:
                population = float(row[chosen_label])

        results.append(PHResult(ph=ph, population=population, lowest_energy_snapshot=snapshot))
    return results


def _analyze_remd_results(
    results: list[tuple[float, ...]],
    remd: ConstantPHRemd,
    cph: "ConstantPH",
    output_path: Path,
    titratable_indices: list[int],
    sample_interval_ns: float,
    *,
    verbose: bool = True,
    skip_pka: bool = False,
    ligand_path: str | None = None,
    run_mmgbsa: bool = False,
    mmgbsa_n_closest_waters: int = 5,
    protonation_swap_steps: int = 1,
    ph_ladder: list[float] | None = None,
    cph_config: ConstantpHSettings | None = None,
    ligand_variant_molecules: "list[Molecule] | None" = None,
) -> dict[str, Any]:
    """Analyze REMD results pooled across all replicas."""
    n_replicas = len(remd.replicas)
    replica_dfs = [
        pd.DataFrame(
            [results[i] for i in range(replica_i, len(results), n_replicas)],
            columns=["ph", *titratable_indices],
        )
        for replica_i in range(n_replicas)
    ]
    combined_df = pd.concat(replica_dfs, ignore_index=True)

    replica_overlays = None
    if not skip_pka:
        per_replica_timeseries = [
            (
                f"replica_{replica_i}",
                compute_pka_timeseries(
                    replica_df,
                    remd.replicas[replica_i],
                    sample_interval_ns=sample_interval_ns,
                ),
            )
            for replica_i, replica_df in enumerate(replica_dfs)
        ]
        replica_overlays = build_replica_overlay_timeseries(
            replica_dfs,
            per_replica_timeseries,
            cph,
            sample_interval_ns=sample_interval_ns,
        )

    analysis = analyze_cph_results(
        combined_df,
        cph,
        output_path,
        sample_interval_ns=sample_interval_ns,
        replica_dfs=replica_dfs,
        overlay_timeseries=replica_overlays or None,
        verbose=verbose,
        skip_pka=skip_pka,
    )

    mmgbsa: dict[str, Any] = {}
    frame_records: list[dict] = []
    if run_mmgbsa and ligand_path is not None:
        logger.info("Computing per-replica MMGBSA")
        mmgbsa = compute_replica_mmgbsa(
            output_path,
            ligand_path,
            replica_dfs,
            protonation_swap_steps=protonation_swap_steps,
            ph_ladder=ph_ladder,
            n_closest_waters=mmgbsa_n_closest_waters,
        )
        frame_records = mmgbsa.pop("mmgbsa_frames", [])
    elif cph_config is not None and ph_ladder:
        logger.info("Computing per-frame implicit-solvent energies")
        frame_records = _per_frame_system_energies(
            output_path, replica_dfs, cph_config, ph_ladder, protonation_swap_steps
        )

    ph_results = _build_ph_results(
        analysis,
        frame_records,
        ph_ladder or [],
        cph=cph,
        ligand_variant_molecules=ligand_variant_molecules,
    )

    return {
        **analysis,
        **mmgbsa,
        "ph_results": ph_results,
    }


def run_cph(
    protein: str | Path,
    output: str | Path,
    ligand: str | Path | None = None,
    cofactor: str | Path | None = None,
    config: ConstantpHRunSettings | None = None,
    *,
    weights: list[float] | None = None,
    resume: bool = False,
    checkpoint_every: int = 50,
    skip_md: bool = False,
) -> dict[str, Any]:
    """Run constant-pH MD for a protein, optionally with ligand and/or cofactor.

    When ``resume=True``, skip system parameterisation/equilibration if
    ``equilibrated.pdb`` and ``run_manifest.json`` exist under ``output``, and
    continue production from ``checkpoints/`` when present.

    When ``skip_md=True``, skip production MD entirely and re-analyse existing
    checkpoints and trajectories under ``output``.
    """
    if config is None:
        config = ConstantpHRunSettings()
    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    protein = Path(protein).expanduser().resolve()
    ligand = Path(ligand).expanduser().resolve() if ligand is not None else None
    cofactor = Path(cofactor).expanduser().resolve() if cofactor is not None else None

    if skip_md:
        resume = True

    integrator_step_size_ps = config.integrator_step_size.value_in_unit(unit.picosecond)
    protonation_swap_interval_ps = config.protonation_swap_interval.value_in_unit(unit.picosecond)
    phs = [config.ph] if isinstance(config.ph, float) else list(config.ph)
    skip_pka = len(phs) == 1
    ckpt_dir = checkpoint_dir(output_path)
    unipka = UnipKa()

    resume_setup = resume and equilibrated_pdb_path(output_path).exists()
    if resume and not resume_setup:
        resume = False

    ligand_setup: LigandSetup | None = None
    cofactor_setup: LigandSetup | None = None
    titratable_residue_indices: list[int]

    if resume_setup:
        manifest = read_run_manifest(output_path)
        validate_run_manifest(
            manifest,
            protein=protein,
            ligand=ligand,
            cofactor=cofactor,
            titratable_residue_indices=None,
            titratable_residue_query=config.titratable_residue_query,
            phs=phs,
            n_replicas=config.n_replicas,
            integrator_step_size_ps=integrator_step_size_ps,
            protonation_swap_interval_ps=protonation_swap_interval_ps,
            cph_config=config.cph_config,
        )
        titratable_residue_indices = manifest["titratable_residue_indices"]
        logger.info("Resuming from equilibrated system on disk")

        if ligand is not None or manifest.get("ligand") is not None:
            ligand_setup = _load_or_build_small_molecule_variants(
                ligand or manifest.get("ligand"),
                output_path / "ligands.csv",
                output_path,
                config,
                unipka,
                residue_name="LIG",
            )
        if cofactor is not None or manifest.get("cofactor") is not None:
            cofactor_setup = _load_or_build_small_molecule_variants(
                cofactor or manifest.get("cofactor"),
                output_path / "cofactors.csv",
                output_path,
                config,
                unipka,
                residue_name="COF",
            )

        equilibrated = PDBFile(str(equilibrated_pdb_path(output_path)))
        omm_top, omm_pos = equilibrated.topology, equilibrated.positions
        system_pdb = output_path / "system.pdb"
    else:
        if ligand is not None:
            ligand_setup = _build_small_molecule_variants(
                ligand,
                config,
                output_path,
                unipka,
                residue_name="LIG",
                csv_name="ligands.csv",
            )
        if cofactor is not None:
            cofactor_setup = _build_small_molecule_variants(
                cofactor,
                config,
                output_path,
                unipka,
                residue_name="COF",
                csv_name="cofactors.csv",
            )

        protein_pdb = PDBFile(str(protein))
        protein_modeller = Modeller(protein_pdb.topology, protein_pdb.positions)

        small_molecules: list[tuple[Molecule, str]] = []
        if ligand_setup is not None:
            small_molecules.append((ligand_setup.variant_molecules[0], "LIG"))
        if cofactor_setup is not None:
            small_molecules.append((cofactor_setup.variant_molecules[0], "COF"))

        if small_molecules:
            omm_top, omm_pos, forcefield = prepare_system(
                protein_modeller=protein_modeller,
                small_molecules=small_molecules,
                padding=1.0,
            )
        else:
            omm_top, omm_pos, forcefield = prepare_protein(
                protein_modeller,
                padding=1.0,
            )

        for res in omm_top.residues():
            res.id = str(res.id)
        # Solvent-free copy: the full solvated box has >9999 residues, whose
        # hex-encoded PDB residue numbers make RDKit's parser fail. This PDB is
        # only used for RDSL titratable-residue selection, which never touches
        # solvent. See write_solute_pdb for details.
        system_pdb = output_path / "system.pdb"
        write_solute_pdb(omm_top, omm_pos, system_pdb)

        residue_reference_dict = generate_residue_reference_dict(
            config.cph_config,
            ligands=_ligand_entries_from_setups(ligand_setup, cofactor_setup) or None,
        )

        titratable_residue_indices = select_titratable_residue_indices(
            omm_top,
            system_pdb,
            residue_reference_dict,
            config.titratable_residue_query,
        )

        if config.restrict_titratable_to_near_ph:
            # Universe of titratable residues (references matched, disulfides
            # dropped); PROPKA hits are matched within it.
            titratable_universe = set(
                select_titratable_residue_indices(omm_top, system_pdb, residue_reference_dict, None)
            )
            try:
                propka_indices = _propka_titratable_indices(
                    protein,
                    omm_top,
                    titratable_universe,
                    phs,
                    protonation_penalty=config.protonation_penalty,
                    temperature=config.cph_config.temperature,
                )
            except Exception as exc:  # PROPKA is best-effort; keep the base set
                logger.warning(
                    f"PROPKA titratable-residue selection failed ({exc}); "
                    "keeping the base selection"
                )
            else:
                if not propka_indices:
                    # Most likely a structure/numbering mismatch rather than a
                    # genuinely inert protein; pruning to nothing would be worse.
                    logger.warning(
                        f"PROPKA flagged no protein residues near the pH ladder "
                        f"{phs}; keeping the base selection rather than pruning it away"
                    )
                else:
                    # Keep only residues that are both requested and near-pKa, but
                    # always retain the ligand/cofactor (PROPKA only sees protein).
                    small_molecule_indices = {
                        r.index for r in omm_top.residues() if r.name in ("LIG", "COF")
                    }
                    keep = propka_indices | small_molecule_indices
                    removed = set(titratable_residue_indices) - keep
                    if removed:
                        removed_names = ", ".join(
                            residue_label(r)
                            for r in omm_top.residues()
                            if r.index in removed
                        )
                        logger.info(
                            f"PROPKA pruned {len(removed)} titratable residue(s) far "
                            f"from the pH ladder {phs}: {removed_names}"
                        )
                    titratable_residue_indices = sorted(set(titratable_residue_indices) & keep)

        if not titratable_residue_indices:
            raise RuntimeError("No titratable residues were found in the supplied topology")

        _log_titratable_residues(omm_top, titratable_residue_indices)

        omm_top, omm_pos = equilibrate(
            omm_top,
            omm_pos,
            forcefield,
            config=config.equilibration_config,
        )
        with equilibrated_pdb_path(output_path).open("w") as pdb_file:
            PDBFile.writeFile(omm_top, omm_pos, pdb_file, keepIds=True)

        residues_by_index = {r.index: r for r in omm_top.residues()}
        write_run_manifest(
            output_path,
            protein=protein,
            ligand=ligand,
            cofactor=cofactor,
            titratable_residue_indices=titratable_residue_indices,
            titratable_residue_labels=[
                residue_label(residues_by_index[i]) for i in titratable_residue_indices
            ],
            titratable_residue_query=config.titratable_residue_query,
            phs=phs,
            n_replicas=config.n_replicas,
            integrator_step_size_ps=integrator_step_size_ps,
            protonation_swap_interval_ps=protonation_swap_interval_ps,
            cph_config=config.cph_config,
            weights=weights,
            weight_equilibration_done=False,
        )

    if resume_setup:
        residue_reference_dict = generate_residue_reference_dict(
            config.cph_config,
            ligands=_ligand_entries_from_setups(ligand_setup, cofactor_setup) or None,
        )
        _log_titratable_residues(omm_top, titratable_residue_indices)

    logger.info(f"pHs: {phs}")
    if skip_pka:
        logger.info("Single pH run: skipping pKa analysis")

    # Molecules registered on the constant-pH force field: every sampled
    # protonation variant plus, when present, each union super-template so
    # ConstantPH can parametrise the union master topology. This list is for
    # SMIRNOFF template registration ONLY - the union must never enter the
    # per-variant INDEXING lists (``setup.variant_molecules``) that
    # ``_analyze_remd_results`` maps trajectory state labels against.
    variant_molecules: list[Molecule] = []
    for setup in (ligand_setup, cofactor_setup):
        if setup is not None:
            variant_molecules.extend(setup.variant_molecules)
            if setup.union_molecule is not None:
                variant_molecules.append(setup.union_molecule)

    # Seed each residue's starting protonation near equilibrium (PROPKA for the
    # protein, dominant protomer for the ligand) rather than fully protonated.
    # Replica 0 gets that deterministic best guess; replicas 1+ are seeded from
    # each residue's target-pH Boltzmann distribution, so pH-REMD starts with a
    # real spread of macrostates to exchange.
    initial_variant_indices: dict[int, int] | list[dict[int, int]] | None = None
    if config.seed_initial_protonation:
        seed_ph = phs[len(phs) // 2]
        deterministic_seed = _seed_initial_variant_indices(
            omm_top,
            residue_reference_dict,
            titratable_residue_indices,
            protein,
            seed_ph,
            ligand_setup,
            cofactor_setup,
        )
        if deterministic_seed:
            residues = list(omm_top.residues())
            seeded = []
            for res_index, variant_index in sorted(deterministic_seed.items()):
                ref = residue_reference_dict.get(residues[res_index].name)
                variant_name = (
                    ref.variant_names[variant_index]
                    if ref is not None and 0 <= variant_index < len(ref.variant_names)
                    else str(variant_index)
                )
                seeded.append(f"{residue_label(residues[res_index])} -> {variant_name}")
            logger.info(f"Seeded replica-0 protonation (pH {seed_ph}): {', '.join(seeded)}")
            initial_variant_indices = _replica_initial_variant_indices(
                deterministic_seed,
                omm_top,
                residue_reference_dict,
                titratable_residue_indices,
                seed_ph,
                config.cph_config.temperature,
                config.n_replicas,
            )


    remd = ConstantPHRemd(
        topology=omm_top,
        positions=omm_pos,
        ph=phs,
        config=config.cph_config,
        references=residue_reference_dict,
        titratable_residue_indices=titratable_residue_indices,
        n_replicas=config.n_replicas,
        ligand_variant_molecules=variant_molecules or None,
        weights=weights,
        initial_variant_indices=initial_variant_indices,
    )
    persist_ligand_setups(output_path, ligand_setup, cofactor_setup)
    cph = remd.replicas[0]
    _apply_barostat(remd, config)

    resume_production = resume and (ckpt_dir / REMD_STATE_FILENAME).exists()
    remd_swap_interval_ps = config.remd_swap_interval.value_in_unit(unit.picosecond)

    if skip_md:
        if not resume_setup:
            raise ValueError("skip_md requires equilibrated.pdb and run_manifest.json under output")
        if not (ckpt_dir / REMD_STATE_FILENAME).exists():
            raise ValueError(f"skip_md requires {ckpt_dir / REMD_STATE_FILENAME}")
        if not (ckpt_dir / PRODUCTION_STATE_FILENAME).exists():
            raise ValueError(f"skip_md requires {ckpt_dir / PRODUCTION_STATE_FILENAME}")
        remd.load_checkpoint(ckpt_dir)
        prod_state = read_production_state(ckpt_dir)
        results = list(prod_state["results"])
        if not results:
            raise ValueError("skip_md requires non-empty production results on disk")
        logger.info(
            f"skip_md: loaded {len(results)} result rows from checkpoint; skipping production MD"
        )
    elif resume_production:
        remd.load_checkpoint(ckpt_dir)
        prod_state = read_production_state(ckpt_dir)
        start_batch = prod_state["batches_completed"]
        results = list(prod_state["results"])
        next_remd_swap_ps = prod_state["next_remd_swap_ps"]
        if next_remd_swap_ps is None:
            next_remd_swap_ps = remd_swap_interval_ps
        logger.info(
            f"Resuming production from batch {start_batch}; "
            f"replica steps: "
            f"{[r.simulation.currentStep for r in remd.replicas]}"
        )
    else:
        manifest = read_run_manifest(output_path) if resume_setup else {}
        if resume_setup and manifest.get("weight_equilibration_done"):
            saved_weights = manifest.get("weights") or weights
            if saved_weights is None:
                raise ValueError(
                    "Manifest marks weight equilibration complete but no weights "
                    "were saved; pass weights explicitly or rerun without resume"
                )
            cur_weights = np.array(saved_weights)
            logger.info(f"Using saved weights from manifest: {cur_weights}")
        else:
            cur_weights = _optimize_remd_weights(remd)
            update_manifest_weights(output_path, list(cur_weights))

        remd.set_weights(cur_weights)
        remd.reset_stats()
        remd.save_checkpoint(ckpt_dir)
        start_batch = 0
        results = []
        next_remd_swap_ps = remd_swap_interval_ps
        logger.info(
            f"pH-REMD with {config.n_replicas} replicas across pHs: {phs}; "
            f"initial replica pHs: {remd.current_ph_values()}"
        )

    num_steps = int(config.production_time / config.integrator_step_size)
    reporter_interval = int(config.reporter_interval / config.integrator_step_size)
    protonation_swap_steps = int(config.protonation_swap_interval / config.integrator_step_size)
    num_batches = num_steps // protonation_swap_steps
    titratable_indices = list(cph.titrations.keys())

    ps_per_batch = protonation_swap_interval_ps
    sample_interval_ns = config.protonation_swap_interval.value_in_unit(unit.nanosecond)

    if not skip_md:
        traj_dir = trajectories_dir(output_path)
        traj_dir.mkdir(parents=True, exist_ok=True)
        trajectory_managers = [
            StateSplitTrajectoryManager(
                replica,
                replica_trajectory_base(output_path, replica_i),
                report_interval=reporter_interval,
                append=resume_production,
            )
            for replica_i, replica in enumerate(remd.replicas)
        ]
        for replica, manager in zip(remd.replicas, trajectory_managers, strict=True):
            replica.simulation.reporters.append(manager)

        with tqdm(
            total=num_batches,
            initial=start_batch,
            desc="Production",
            unit="frames",
        ) as pbar:
            for i in range(start_batch, num_batches):
                iter_start = time.time()
                remd.step(protonation_swap_steps)
                remd.attempt_mc_step()
                current_time_ps = (i + 1) * ps_per_batch
                if config.use_ph_remd and current_time_ps >= next_remd_swap_ps:
                    remd.attempt_adjacent_exchanges()
                    next_remd_swap_ps += remd_swap_interval_ps
                results.extend(remd.replica_state_vectors())

                if checkpoint_every > 0 and (i + 1) % checkpoint_every == 0:
                    remd.save_checkpoint(ckpt_dir)
                    write_production_state(
                        ckpt_dir,
                        batches_completed=i + 1,
                        next_remd_swap_ps=next_remd_swap_ps,
                        results=results,
                    )

                    _analyze_remd_results(
                        results,
                        remd,
                        cph,
                        output_path,
                        titratable_indices,
                        sample_interval_ns,
                        verbose=False,
                        skip_pka=skip_pka,
                        ligand_variant_molecules=(
                            ligand_setup.variant_molecules if ligand_setup else None
                        ),
                    )

                iter_time = time.time() - iter_start
                sim_time_ns = ps_per_batch / 1000.0
                ns_per_day = (
                    config.n_replicas * sim_time_ns * 86400 / iter_time if iter_time > 0 else 0
                )
                pbar.set_postfix(
                    Time=f"{current_time_ps:.0f}ps",
                    ns_per_day=f"{ns_per_day:.2f}",
                )
                pbar.update(1)

        remd.save_checkpoint(ckpt_dir)
        write_production_state(
            ckpt_dir,
            batches_completed=num_batches,
            next_remd_swap_ps=next_remd_swap_ps,
            results=results,
        )

        for manager in trajectory_managers:
            manager.close()

    ligand_path_for_mmgbsa = ligand
    if ligand_path_for_mmgbsa is None and resume_setup:
        ligand_path_for_mmgbsa = read_run_manifest(output_path).get("ligand")

    analysis = _analyze_remd_results(
        results,
        remd,
        cph,
        output_path,
        titratable_indices,
        sample_interval_ns,
        verbose=True,
        skip_pka=skip_pka,
        ligand_path=ligand_path_for_mmgbsa,
        run_mmgbsa=ligand_path_for_mmgbsa is not None,
        mmgbsa_n_closest_waters=config.mmgbsa_n_closest_waters,
        protonation_swap_steps=protonation_swap_steps,
        ph_ladder=phs,
        cph_config=config.cph_config,
        ligand_variant_molecules=(ligand_setup.variant_molecules if ligand_setup else None),
    )

    return {
        **analysis,
        "cph": cph,
        "remd": remd,
        "replicas": remd.replicas,
    }


if __name__ == "__main__":
    protein_pdb_path = "/Users/finlaymaclean/Desktop/mtx.pdb"
    ligand_path = "/Users/finlaymaclean/Desktop/mtx.sdf"
    # ligand_path = None
    weights = None

    run_cph(
        protein_pdb_path,
        output="~/Desktop/cph_holo",
        ligand=ligand_path,
        resume=True,
        skip_md=False,
        config=ConstantpHRunSettings(
            titratable_residue_query="(resn ASP and resi 27) or (resn LIG)",
            ph=7.0,
            n_replicas=1,
            weights=weights,
            production_time=500 * unit.picosecond,
        ),
    )
