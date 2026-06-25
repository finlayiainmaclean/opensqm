import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mdtraj as md
import numpy as np
import pandas as pd
from loguru import logger
from openff.toolkit.topology import Molecule  # type: ignore
from openmm import Context, LangevinIntegrator, MonteCarloBarostat, unit, System
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
from opensqm.cph.constantph import select_titratable_residue_indices
from opensqm.cph.mmgbsa import compute_replica_mmgbsa
from opensqm.cph.ph_remd import ConstantPHRemd
from opensqm.cph.pka import (
    analyze_cph_results,
    build_replica_overlay_timeseries,
    compute_pka_timeseries,
)
from opensqm.cph.reference_energy import (
    build_protonation_states,
    generate_residue_reference_dict,
)
from opensqm.cph.simulation_config import ConstantpHSettings
from opensqm.cph.mmgbsa import _batch_index_for_step, _snap_ph
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


def _default_equilibration_config() -> EquilibrationSettings:
    return EquilibrationSettings(
        npt_time=100 * unit.picoseconds,
        warmup_time=10 * unit.picoseconds,
    )



class ConstantpHRunSettings(BaseModel):
    """Settings for a constant-pH MD run."""

    model_config = ConfigDict(frozen=True)
    # Equilibration settings
    equilibration_config: EquilibrationSettings = Field(default_factory=_default_equilibration_config)
    # Production settings
    production_time: OpenMMQuantity[unit.nanosecond] = 10 * unit.nanosecond
    integrator_step_size: OpenMMQuantity[unit.picosecond] = 0.004 * unit.picoseconds
    barostat_pressure: OpenMMQuantity[unit.bar] | None = None
    reporter_interval: OpenMMQuantity[unit.picosecond] = 1 * unit.picosecond
    # CPH settings
    cph_config: ConstantpHSettings = Field(default_factory=ConstantpHSettings)
    titratable_residue_query: str | None = None
    ligand_terminal_ring_mc: bool = False
    mmgbsa_n_closest_waters: int = 5
    ligand_protonation: bool = True
    protonation_swap_interval: OpenMMQuantity[unit.picosecond] = 0.2 * unit.picoseconds
    remd_swap_interval: OpenMMQuantity[unit.picosecond] = 10 * unit.picoseconds
    use_ph_remd: bool = True
    n_replicas: int = 3
    pH: float | list[float] = Field(default_factory=lambda: [1.0, 7.0, 14.0])


@dataclass(frozen=True)
class LigandSetup:
    variant_molecules: list[Molecule]
    transitions: list[Any]
    ring_flip_bonds: list[tuple[str, str]]


# pKa window for ligand titration-site enumeration. Only sites whose
# micro-pKa lies in this range are titrated; the rest are held fixed at the
# input protomer's protonation state. Bracketing physiological pH captures the
# sites that actually titrate in a near-neutral constant-pH run.
_LIGAND_PKA_WINDOW = (6.0, 8.0)


def _protomer_key(mol: Chem.Mol) -> str:
    """FixedH InChIKey that distinguishes protonation states.

    Mirrors ``unipka._same_mol`` so the protomers returned by
    ``get_microstates`` can be deduplicated and matched back to the variant
    list by chemical identity (atom names/coordinates are ignored).
    """
    return Chem.MolToInchiKey(Chem.RemoveHs(Chem.Mol(mol)), options="/FixedH")


def _variants_from_microstates(
    molecule_rdmol: Chem.Mol,
    microstates: list,
) -> tuple[list[Chem.Mol], list[tuple[int, int, float]]]:
    """Turn unipka microstates into aligned protomers + a transition graph.

    Each :class:`unipka.Microstate` is a single-site, one-proton acid/base
    pair with its own micro-pKa, all enumerated relative to ``molecule_rdmol``.
    The variant set is therefore a *star* rooted at the input protomer, so the
    transition graph is connected and one-proton-stepped by construction -- no
    ladder assembly is needed. Returns ``(molecule_mols, transitions)`` where
    ``molecule_mols`` are pose-aligned RDKit mols (most-protonated first, so
    ``molecule_mols[0]`` seeds the production topology) and ``transitions`` are
    ``(parent_index, child_index, pka)`` tuples indexing into them with
    ``parent`` the acid and ``child`` the conjugate base.
    """
    if not microstates:
        return [molecule_rdmol], []

    keys = [_protomer_key(molecule_rdmol)]
    state_mols = [molecule_rdmol]
    for ms in microstates:
        for partner in (ms.acid, ms.base):
            key = _protomer_key(partner)
            if key not in keys:
                keys.append(key)
                state_mols.append(partner)

    # Order by formal charge descending so the most-protonated state is first,
    # matching the prior charge-sorted convention the production topology and
    # super-template construction rely on.
    order = sorted(
        range(len(state_mols)),
        key=lambda i: Chem.GetFormalCharge(state_mols[i]),
        reverse=True,
    )
    state_mols = [state_mols[i] for i in order]
    keys = [keys[i] for i in order]
    key_to_idx = {key: i for i, key in enumerate(keys)}

    molecule_mols = build_protonation_states(state_mols, geometry_mol=molecule_rdmol)
    transitions = [
        (
            key_to_idx[_protomer_key(ms.acid)],
            key_to_idx[_protomer_key(ms.base)],
            float(ms.pka),
        )
        for ms in microstates
    ]
    return molecule_mols, transitions


def _ligand_transitions_path(output_path: Path, csv_name: str) -> Path:
    return output_path / f"{Path(csv_name).stem}_transitions.json"


def _save_ligand_transitions(
    output_path: Path, csv_name: str, transitions: list[tuple[str, str, float]],
) -> None:
    _ligand_transitions_path(output_path, csv_name).write_text(
        json.dumps(
            [{"parent": p, "child": c, "pka": pka} for p, c, pka in transitions],
            indent=2,
        )
    )


def _load_ligand_transitions(
    output_path: Path, csv_name: str,
) -> list[tuple[str, str, float]] | None:
    path = _ligand_transitions_path(output_path, csv_name)
    if not path.exists():
        return None
    return [
        (e["parent"], e["child"], float(e["pka"]))
        for e in json.loads(path.read_text())
    ]


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
        lo, hi = _LIGAND_PKA_WINDOW
        microstates = unipka.get_microstates(molecule_rdmol, min_pka=lo, max_pka=hi)
        molecule_mols, index_transitions = _variants_from_microstates(
            molecule_rdmol, microstates,
        )
        logger.info(
            f"{residue_name}: {len(molecule_mols)} protonation states from "
            f"{len(microstates)} titratable site(s) in pKa [{lo}, {hi}]"
        )
    else:
        molecule_mols = [molecule_rdmol]
        index_transitions = []

    variant_molecules: list[Molecule] = []
    for i, rdmol in enumerate(molecule_mols):
        offmol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
        offmol.name = residue_name if i == 0 else f"{residue_name}{i}"
        variant_molecules.append(offmol)

    # Resolve the index-based transition graph to variant names now that the
    # molecules are named (NamedTransition is what generate_ligand_reference
    # consumes), and persist it so resumed runs need not re-query unipka.
    transitions = [
        (variant_molecules[parent].name, variant_molecules[child].name, pka)
        for parent, child, pka in index_transitions
    ]

    pd.DataFrame(
        {
            "residue_name": [mol.name for mol in variant_molecules],
            "smiles": [Chem.MolToSmiles(mol) for mol in molecule_mols],
        }
    ).to_csv(output_path / csv_name, index=False)
    _save_ligand_transitions(output_path, csv_name, transitions)

    if config.ligand_terminal_ring_mc:
        ring_flip_bonds = autodetect_flip_dihedrals_named(molecule_mols[0])
        logger.info(f"{residue_name} ring_flip_bonds: {ring_flip_bonds}")
    else:
        ring_flip_bonds = []

    return LigandSetup(variant_molecules, transitions, ring_flip_bonds)


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
    setup = _reload_ligand_setup_from_csv(csv_path, output_path, config)
    if setup is not None:
        return setup
    logger.info(
        f"{csv_path.name} not found; rebuilding {residue_name} variants from file"
    )
    return _build_small_molecule_variants(
        molecule_path, config, output_path, unipka,
        residue_name=residue_name, csv_name=csv_path.name,
    )


def _reload_ligand_setup_from_csv(
    csv_path: Path,
    output_path: Path,
    config: ConstantpHRunSettings,
) -> LigandSetup | None:
    """Rebuild ligand/cofactor setup from a prior run's variant CSV.

    Returns ``None`` (triggering a full rebuild) when either the variant CSV
    or its sibling transitions JSON is missing -- e.g. for runs created before
    the transition graph was persisted.
    """
    if not csv_path.exists():
        return None
    transitions = _load_ligand_transitions(csv_path.parent, csv_path.name)
    if transitions is None:
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

    return LigandSetup(variant_molecules, transitions, ring_flip_bonds)


def _ligand_entries_from_setups(
    ligand_setup: LigandSetup | None,
    cofactor_setup: LigandSetup | None,
) -> list[tuple[list[Molecule], list[Any], list[tuple[str, str]]]]:
    entries = []
    for setup in (ligand_setup, cofactor_setup):
        if setup is None:
            continue
        entries.append(
            (setup.variant_molecules, setup.transitions, setup.ring_flip_bonds)
        )
    return entries


def _log_titratable_residues(topology, titratable_residue_indices: list[int]) -> None:
    residue_names = np.array([r.name for r in topology.residues()])
    formatted = ", ".join(
        f"{residue_names[i]} ({i})" for i in titratable_residue_indices
    )
    logger.info(f"Titratable residues: {formatted}")


def _apply_barostat(remd: ConstantPHRemd, config: ConstantpHRunSettings) -> None:
    if config.barostat_pressure is None:
        return
    for replica in remd.replicas:
        replica.simulation.system.addForce(
            MonteCarloBarostat(config.barostat_pressure, 300 * unit.kelvin)
        )
        replica.simulation.context.reinitialize(preserveState=True)


def _optimize_remd_weights(remd: ConstantPHRemd) -> np.ndarray:
    cur_weights = np.array(remd.weights)
    logger.info("Optimising weights")
    num_successful_swaps = 0
    equilibration_mc_steps = 2000
    with tqdm(range(equilibration_mc_steps), desc="Equilibration") as pbar:
        for _ in pbar:
            prev_weights = np.array(cur_weights)
            remd.step(50)  # 0.2ps
            swaps = remd.attemptMCStep()
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
    topology: Topology                      # box vectors set to this frame's cell
    positions: unit.Quantity                # (n_atoms, 3), physical atoms of that protonation state
    system: System | None = None
    ligand: "Chem.Mol | None" = None


@dataclass
class pHResult:
    """Per-pH summary for downstream use (e.g. modbinddg)."""
    ph: float
    population: float                       # joint population fraction of the snapshot's state at this pH
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
        traj_csv = pd.read_csv(replica_trajectory_index(output_path, replica_i))
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
                zip(state_rows["frame_ix"].astype(int), state_rows["time_ns"])
            )
            frame_to_step = dict(
                zip(state_rows["frame_ix"].astype(int), state_rows["step"].astype(int))
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
                records.append({
                    "ph": ph,
                    "state_label": state_label,
                    "time_ns": frame_to_time[frame_ix],
                    "replica_i": replica_i,
                    "energy": energy,
                    "energy_type": "potential",
                    "dcd_path": dcd_path,
                    "pdb_path": pdb_path,
                    "frame_ix": frame_ix,
                })
    return records


def _ligand_rdmol_for_state(
    cph,
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
    ligand_keys = [
        k for k in keys if str(resname_by_index.get(k, "")).startswith(ligand_resname)
    ]
    if len(ligand_keys) != 1:
        return None
    variant_index = indices[keys.index(ligand_keys[0])]
    if not 0 <= variant_index < len(ligand_variant_molecules):
        return None
    return ligand_variant_molecules[variant_index].to_rdkit()


def _state_label_to_joint_label(cph, state_label: str) -> "str | None":
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
    for res_idx, state in zip(titratable_indices, indices):
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
    cph=None,
    ligand_variant_molecules: "list[Molecule] | None" = None,
) -> list[pHResult]:
    """Build per-pH summaries: the dominant microstate's population and a
    representative (lowest-energy) frame drawn from that microstate."""
    joint_pops_df: pd.DataFrame = analysis.get("joint_populations", pd.DataFrame())

    if frame_records:
        valid_times = [r["time_ns"] for r in frame_records if not np.isnan(r["time_ns"])]
        cutoff = max(valid_times) * (1.0 - last_fraction) if valid_times else float("inf")
        tail = [r for r in frame_records if r["time_ns"] >= cutoff]
    else:
        tail = []

    results: list[pHResult] = []
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
                r for r in ph_tail
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
            frame = md.load_frame(str(best["dcd_path"]), best["frame_ix"], top=str(best["pdb_path"]))
            topology = PDBFile(str(best["pdb_path"])).topology
            if frame.unitcell_vectors is not None:
                topology.setPeriodicBoxVectors(frame.unitcell_vectors[0] * unit.nanometer)
            snapshot = SystemState(
                topology=topology,
                positions=frame.xyz[0] * unit.nanometer,
                ligand=_ligand_rdmol_for_state(
                    cph, ligand_variant_molecules, best["state_label"]
                ),
            )
            # Report the population of the microstate the chosen frame is in.
            chosen_label = _state_label_to_joint_label(cph, best["state_label"])
            if row is not None and chosen_label is not None and chosen_label in row.index:
                population = float(row[chosen_label])

        results.append(pHResult(ph=ph, population=population, lowest_energy_snapshot=snapshot))
    return results


def _analyze_remd_results(
    results: list[tuple[float, ...]],
    remd: ConstantPHRemd,
    cph,
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
        "replica_results": [
            [results[i] for i in range(replica_i, len(results), n_replicas)]
            for replica_i in range(n_replicas)
        ],
        "ph_results": ph_results,
    }


def run_cph(
    protein: str | Path,
    output: str | Path,
    ligand: str | Path | None = None,
    cofactor: str | Path | None = None,
    config: ConstantpHRunSettings = ConstantpHRunSettings(),
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
    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    protein = Path(protein).expanduser().resolve()
    ligand = Path(ligand).expanduser().resolve() if ligand is not None else None
    cofactor = Path(cofactor).expanduser().resolve() if cofactor is not None else None


    if skip_md:
        resume = True

    integrator_step_size_ps = config.integrator_step_size.value_in_unit(unit.picosecond)
    protonation_swap_interval_ps = config.protonation_swap_interval.value_in_unit(
        unit.picosecond
    )
    pHs = [config.pH] if isinstance(config.pH, float) else list(config.pH)
    skip_pka = len(pHs) == 1
    ckpt_dir = checkpoint_dir(output_path)
    unipka = UnipKa()

    resume_setup = resume and equilibrated_pdb_path(output_path).exists()
    if resume and not resume_setup:
        resume=False

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
            pHs=pHs,
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
                output_path, config, unipka,
                residue_name="LIG",
            )
        if cofactor is not None or manifest.get("cofactor") is not None:
            cofactor_setup = _load_or_build_small_molecule_variants(
                cofactor or manifest.get("cofactor"),
                output_path / "cofactors.csv",
                output_path, config, unipka,
                residue_name="COF",
            )

        equilibrated = PDBFile(str(equilibrated_pdb_path(output_path)))
        omm_top, omm_pos = equilibrated.topology, equilibrated.positions
        system_pdb = output_path / "system.pdb"
    else:
        if ligand is not None:
            ligand_setup = _build_small_molecule_variants(
                ligand, config, output_path, unipka,
                residue_name="LIG", csv_name="ligands.csv",
            )
        if cofactor is not None:
            cofactor_setup = _build_small_molecule_variants(
                cofactor, config, output_path, unipka,
                residue_name="COF", csv_name="cofactors.csv",
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
                protein_modeller, padding=1.0,
            )

        for res in omm_top.residues():
            res.id = str(res.id)
        system_pdb = output_path / "system.pdb"
        with system_pdb.open("w") as pdb_file:
            PDBFile.writeFile(omm_top, omm_pos, pdb_file, keepIds=True)

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

        if not titratable_residue_indices:
            raise RuntimeError(
                "No titratable residues were found in the supplied topology"
            )

        _log_titratable_residues(omm_top, titratable_residue_indices)

        omm_top, omm_pos = equilibrate(
            omm_top, omm_pos, forcefield, config=config.equilibration_config,
        )
        with equilibrated_pdb_path(output_path).open("w") as pdb_file:
            PDBFile.writeFile(omm_top, omm_pos, pdb_file, keepIds=True)

        write_run_manifest(
            output_path,
            protein=protein,
            ligand=ligand,
            cofactor=cofactor,
            titratable_residue_indices=titratable_residue_indices,
            titratable_residue_query=config.titratable_residue_query,
            pHs=pHs,
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

    logger.info(f"pHs: {pHs}")
    if skip_pka:
        logger.info("Single pH run: skipping pKa analysis")

    variant_molecules: list[Molecule] = []
    for setup in (ligand_setup, cofactor_setup):
        if setup is not None:
            variant_molecules.extend(setup.variant_molecules)

    remd = ConstantPHRemd(
        topology=omm_top,
        positions=omm_pos,
        pH=pHs,
        config=config.cph_config,
        references=residue_reference_dict,
        titratable_residue_indices=titratable_residue_indices,
        n_replicas=config.n_replicas,
        ligand_variant_molecules=variant_molecules or None,
        weights=weights,
    )
    persist_ligand_setups(output_path, ligand_setup, cofactor_setup)
    cph = remd.replicas[0]
    _apply_barostat(remd, config)

    resume_production = resume and (ckpt_dir / REMD_STATE_FILENAME).exists()
    remd_swap_interval_ps = config.remd_swap_interval.value_in_unit(unit.picosecond)

    if skip_md:
        if not resume_setup:
            raise ValueError(
                "skip_md requires equilibrated.pdb and run_manifest.json under output"
            )
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
            f"skip_md: loaded {len(results)} result rows from checkpoint; "
            "skipping production MD"
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
            f"pH-REMD with {config.n_replicas} replicas across pHs: {pHs}; "
            f"initial replica pHs: {remd.current_ph_values()}"
        )

    num_steps = int(config.production_time / config.integrator_step_size)
    reporter_interval = int(config.reporter_interval / config.integrator_step_size)
    protonation_swap_steps = int(
        config.protonation_swap_interval / config.integrator_step_size
    )
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
                remd.attemptMCStep()
                current_time_ps = (i + 1) * ps_per_batch
                if config.use_ph_remd and current_time_ps >= next_remd_swap_ps:
                    remd.attemptAdjacentExchanges()
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
                ns_per_day = config.n_replicas * sim_time_ns * 86400 / iter_time if iter_time > 0 else 0
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
        ph_ladder=pHs,
        cph_config=config.cph_config,
        ligand_variant_molecules=(
            ligand_setup.variant_molecules if ligand_setup else None
        ),
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


        config=ConstantpHRunSettings(titratable_residue_query="(resn ASP and resi 27) or (resn LIG)",
        pH=7.0, n_replicas=3, weights=weights)
    )
