"""CTMD: hit triage by the reversible work needed to unseat a docked pose.

Command-line entry point for the protocol of Shekhar et al. (bioRxiv 2026).
Equilibrates the complex, runs independent well-tempered metadynamics replicas
that bias the ligand RMSD until the ligand leaves, and reports the c(t) at which
it left. Unlike ModBinddG there is no unbound state to prepare: the score is a
property of the bound trajectory alone, which is what makes this cheap enough to
triage a screen.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import click
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath, CloudPath
from loguru import logger
from openmm import unit
from rdkit import RDLogger
from tqdm import tqdm

from opensqm.ctmd.config import CTMDSettings
from opensqm.ctmd.metad import CTMDTrajectory, run_ctmd_replica
from opensqm.ctmd.score import bootstrap_score, score_replica
from opensqm.md.platforms import set_platform
from opensqm.md.run_mmgbsa import MMGBSASettings, run_mmgbsa
from opensqm.modbind.states import (
    PreparedState,
    build_bound_state_from_state,
    load_prepared_state,
    save_prepared_state,
)

RDLogger.DisableLog("rdApp.warning")


def _checkpoint_path(checkpoint_dir: Path, index: int) -> Path:
    return checkpoint_dir / f"ctmd_{index:04d}.npz"


def _save_trajectory(path: Path, trajectory: CTMDTrajectory) -> None:
    np.savez(
        path,
        rmsd_nm=trajectory.rmsd_nm,
        ct_kj=trajectory.ct_kj,
        frame_interval_ps=np.asarray(trajectory.frame_interval_ps),
    )


def _load_trajectory(path: Path) -> CTMDTrajectory:
    data = np.load(path, allow_pickle=False)
    return CTMDTrajectory(
        rmsd_nm=data["rmsd_nm"],
        ct_kj=data["ct_kj"],
        frame_interval_ps=float(data["frame_interval_ps"]),
    )


def collect_replicas(
    state: PreparedState,
    config: CTMDSettings,
    *,
    checkpoint_dir: Path,
    trajectory_dir: Path,
    resume: bool = True,
) -> list[CTMDTrajectory]:
    """Run (or resume) the metadynamics replicas, checkpointing each one."""
    # Scoring draws replicas without replacement, so catch an unscoreable run
    # here rather than after spending every replica on it.
    if config.n_replicas < config.n_score_replicas:
        raise ValueError(
            f"n_replicas ({config.n_replicas}) must be at least "
            f"n_score_replicas ({config.n_score_replicas})"
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    trajectories: list[CTMDTrajectory] = []
    with tqdm(total=config.n_replicas, desc="CTMD replicas", unit="replica") as progress:
        for index in range(config.n_replicas):
            checkpoint = _checkpoint_path(checkpoint_dir, index)
            if resume and checkpoint.exists():
                trajectories.append(_load_trajectory(checkpoint))
                logger.info(f"Replica {index}: loaded from cache")
                progress.update(1)
                continue

            trajectory = run_ctmd_replica(
                state,
                config,
                seed=config.random_seed + index + 1,
                dcd_path=str(trajectory_dir / f"ctmd_{index:04d}.dcd"),
            )
            _save_trajectory(checkpoint, trajectory)
            trajectories.append(trajectory)
            ct_value, residence_ns, committed = score_replica(trajectory, config)
            verb = "left at" if committed else "held for"
            logger.info(
                f"Replica {index}: {verb} {residence_ns:.3f} ns, c(t) = {ct_value:.1f} kJ/mol"
            )
            progress.update(1)
    return trajectories


def analyze_ctmd(
    trajectories: list[CTMDTrajectory], config: CTMDSettings, output_path: Path
) -> dict:
    """Score the replicas and write ``results.csv`` and ``replicas.csv``."""
    per_replica = [score_replica(trajectory, config) for trajectory in trajectories]
    ct_values = [value for value, _, _ in per_replica]
    residence_times = [residence for _, residence, _ in per_replica]

    pd.DataFrame(
        {
            "replica": range(len(per_replica)),
            "ct_kj_per_mol": ct_values,
            "residence_time_ns": residence_times,
            "committed": [committed for _, _, committed in per_replica],
            "duration_ns": [trajectory.duration_ns for trajectory in trajectories],
        }
    ).to_csv(output_path / "replicas.csv", index=False)

    # Bare dict: the scores are numeric but config_hash is not, and the repo does
    # not type these result payloads (see modbind.analyze).
    results: dict = bootstrap_score(
        ct_values,
        residence_times,
        n_select=config.n_score_replicas,
        repeats=config.n_bootstrap,
        seed=config.random_seed,
    )
    results["n_replicas"] = len(trajectories)
    results["n_committed"] = sum(committed for _, _, committed in per_replica)
    results["total_sim_time_ns"] = sum(trajectory.duration_ns for trajectory in trajectories)
    results["config_hash"] = config.hash()

    pd.DataFrame([results]).to_csv(output_path / "results.csv", index=False)
    logger.info(
        f"c(t) score = {results['ct_score']:.1f} +/- {results['ct_score_std']:.1f} kJ/mol, "
        f"residence time = {results['residence_time_ns']:.3f} ns "
        f"({results['n_committed']}/{results['n_replicas']} replicas left the pose)"
    )
    return results


def run_ctmd(
    protein: str,
    ligand: str,
    output: str,
    config: CTMDSettings | None = None,
) -> dict:
    """Run CTMD for one protein-ligand pair and return the triage score.

    An MMGBSA protomer funnel equilibrates the complex, picks the ligand
    protonation state in the pocket and supplies the lowest-energy frame as the
    starting pose. That frame is the RMSD reference for every replica.

    ``protein``, ``ligand`` and ``output`` may each be a local path or an
    ``s3://`` URI. A local ``output`` is the working directory: staged inputs,
    the equilibrated state and the replica checkpoints persist there and are
    reused, so a re-run only redoes what is missing.
    """
    if config is None:
        config = CTMDSettings()

    out_dir = AnyPath(output)
    remote = isinstance(out_dir, CloudPath)
    scratch = tempfile.TemporaryDirectory() if remote else None
    work_dir = Path(scratch.name) if scratch is not None else Path(output)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        protein_src, ligand_src = AnyPath(protein), AnyPath(ligand)
        local_protein = work_dir / f"protein_input{protein_src.suffix or '.pdb'}"
        local_ligand = work_dir / f"ligand_input{ligand_src.suffix or '.sdf'}"
        if not local_protein.exists():
            local_protein.write_bytes(protein_src.read_bytes())
        if not local_ligand.exists():
            local_ligand.write_bytes(ligand_src.read_bytes())

        equil_dir = work_dir / "equil"
        scores_path = equil_dir / "mmgbsa_scores.json"
        state = load_prepared_state(equil_dir, "bound")
        if state is not None and scores_path.exists():
            mmgbsa_scores = json.loads(scores_path.read_text())
            logger.info(f"Loaded cached equilibrated state + MMGBSA scores from {equil_dir}")
        else:
            logger.info(
                f"Running MMGBSA protomer-funnel equilibration "
                f"({config.mmgbsa_equilibration_ns} ns production)"
            )
            mmgbsa_result = run_mmgbsa(
                str(local_protein),
                str(local_ligand),
                output=str(work_dir / "mmgbsa_equilibration"),
                config=MMGBSASettings(
                    production_time=config.mmgbsa_equilibration_ns * unit.nanosecond,
                    n_replicas=1,
                    protomer_ph=7.0,
                    protonation_penalty=3.0 * unit.kilocalories_per_mole,
                ),
            )
            mmgbsa_scores = {}
            for key, value in mmgbsa_result.scores.items():
                try:
                    mmgbsa_scores[key] = float(value)
                except (TypeError, ValueError):
                    mmgbsa_scores[key] = value
            # The backbone restraints this attaches are what pin the reference
            # frame, so the lab-frame ligand RMSD is the protein-aligned one.
            state = build_bound_state_from_state(mmgbsa_result.snapshot)
            save_prepared_state(state, equil_dir, "bound")
            scores_path.write_text(json.dumps(mmgbsa_scores))

        trajectories = collect_replicas(
            state,
            config,
            checkpoint_dir=work_dir / "checkpoints",
            trajectory_dir=work_dir / "trajectories",
        )
        results = analyze_ctmd(trajectories, config, work_dir)
        results["mmgbsa_score"] = mmgbsa_scores.get("mmgbsa_score", float("nan"))

        if remote:
            out_dir.mkdir(parents=True, exist_ok=True)
            for name in ("results.csv", "replicas.csv"):
                (out_dir / name).write_bytes((work_dir / name).read_bytes())
            logger.info(f"Published results to {out_dir}")
        else:
            logger.info(f"Saved results to {work_dir / 'results.csv'}")
    finally:
        if scratch is not None:
            scratch.cleanup()

    return results


@click.command()
@click.option("--protein", required=True, help="Protein PDB file (local path or s3:// URI).")
@click.option("--ligand", required=True, help="Ligand MOL/SDF file (local path or s3:// URI).")
@click.option("--output", required=True, help="Output directory (local path or s3:// prefix).")
@click.option("--n-replicas", default=10, show_default=True, help="Metadynamics replicas to run.")
@click.option(
    "--n-score-replicas",
    default=3,
    show_default=True,
    help="Replicas per bootstrap draw; the score is the minimum c(t) over a draw.",
)
@click.option(
    "--max-time",
    default=5.0,
    show_default=True,
    help="Cap on one replica's simulated time (ns).",
)
@click.option(
    "--platform",
    "platform",
    type=click.Choice(["cuda", "mps"], case_sensitive=False),
    default=None,
    help="Force the OpenMM compute platform: 'cuda' (NVIDIA GPU) or 'mps' "
    "(Apple Silicon Metal/OpenCL). Fails if unavailable. "
    "Default: OpenMM auto-selects the fastest platform.",
)
def main(
    protein: str,
    ligand: str,
    output: str,
    n_replicas: int,
    n_score_replicas: int,
    max_time: float,
    platform: str | None,
) -> None:
    """Run CTMD hit triage from the command line."""
    set_platform(platform)
    config = CTMDSettings(
        n_replicas=n_replicas,
        n_score_replicas=n_score_replicas,
        max_time=max_time * unit.nanosecond,
    )
    results = run_ctmd(protein, ligand, output, config=config)
    logger.info(
        f"Done: c(t) = {results['ct_score']:.1f} +/- {results['ct_score_std']:.1f} kJ/mol "
        f"over {results['total_sim_time_ns']:.2f} ns"
    )


if __name__ == "__main__":
    main()
