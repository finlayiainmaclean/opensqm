"""ModBinddG: absolute binding free energy via population reweighting.

Command-line entry point for the two-state population-reweighting method of
Sinko et al. (*PNAS* 2026). Builds the bound (ligand-protein) and unbound
(ligand-in-solvent) states, runs high-temperature escape simulations, and
reports ``ΔG°`` with a bootstrapped confidence interval.
"""

import json
import tempfile
from pathlib import Path

import click
from cloudpathlib import AnyPath, CloudPath
from loguru import logger
from openmm import unit
from rdkit import RDLogger

from opensqm.md.equilibrate import EquilibrationSettings
from opensqm.md.platforms import set_platform
from opensqm.md.run_mmgbsa import MMGBSASettings, run_mmgbsa
from opensqm.modbind.analyze import analyze_modbinddg
from opensqm.modbind.config import ModBindDGSettings
from opensqm.modbind.simulate import collect_trajectories
from opensqm.modbind.states import (
    build_bound_state_from_state,
    build_unbound_state,
    load_prepared_state,
    save_prepared_state,
)

RDLogger.DisableLog("rdApp.warning")

MANIFEST_FILENAME = "modbinddg_manifest.json"


def run_modbind(
    protein: str,
    ligand: str,
    output: str,
    config: ModBindDGSettings | None = None,
) -> dict:
    """Run a full ModBinddG calculation for one protein-ligand pair.

    An MMGBSA protomer funnel is the equilibration step: it selects the ligand
    protonation state in the pocket, equilibrates the solvated complex, runs a
    short production MD, and returns the lowest-energy frame. That frame is the
    starting conformation for the escape simulations.

    ``protein``, ``ligand`` and ``output`` may each be a local path or an
    ``s3://`` URI. When ``output`` is a local directory it is used as the working
    directory: staged inputs, the equilibrated states, escape trajectories and
    results are all written there and REUSED on re-run -- any stage whose
    checkpoint already exists is skipped, so re-running only redoes the analysis.
    For an ``s3://`` output a local temp dir is used and ``results.csv`` published.
    """
    if config is None:
        config = ModBindDGSettings()

    out_dir = AnyPath(output)
    remote = isinstance(out_dir, CloudPath)
    scratch = tempfile.TemporaryDirectory() if remote else None
    work_dir = Path(scratch.name) if scratch is not None else Path(output)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Stage inputs into the work dir, skipping the copy if already present so
        # re-runs are cheap. For a local output these persist with the results.
        protein_src, ligand_src = AnyPath(protein), AnyPath(ligand)
        local_protein = work_dir / f"protein_input{protein_src.suffix or '.pdb'}"
        local_ligand = work_dir / f"ligand_input{ligand_src.suffix or '.sdf'}"
        if not local_protein.exists():
            local_protein.write_bytes(protein_src.read_bytes())
        if not local_ligand.exists():
            local_ligand.write_bytes(ligand_src.read_bytes())

        checkpoint_dir = work_dir / "checkpoints"
        trajectory_dir = work_dir / "trajectories"

        # --- MMGBSA protomer-funnel equilibration (cached) ---
        # Find the bound protomer, equilibrate the solvated complex, run a short
        # production MD, and take the lowest-energy frame as the escape start. The
        # equilibrated states are temperature-independent, so they are cached as
        # CIF + System XML and reused on re-run.
        equil_dir = work_dir / "equil"
        scores_path = equil_dir / "mmgbsa_scores.json"
        bound_state = load_prepared_state(equil_dir, "bound")
        unbound_state = load_prepared_state(equil_dir, "unbound")
        if bound_state is not None and unbound_state is not None and scores_path.exists():
            mmgbsa_scores = json.loads(scores_path.read_text())
            logger.info(f"Loaded cached equilibrated states + MMGBSA scores from {equil_dir}")
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
            snapshot = mmgbsa_result.snapshot
            mmgbsa_scores = {}
            for key, value in mmgbsa_result.scores.items():
                try:
                    mmgbsa_scores[key] = float(value)
                except (TypeError, ValueError):
                    mmgbsa_scores[key] = value
            logger.info(
                f"MMGBSA equilibration score: {mmgbsa_scores['interaction_energy']:.2f} kcal/mol"
            )

            logger.info("Building and equilibrating unbound state")
            unbound_state = build_unbound_state(
                snapshot.ligand,
                equilibration_config=EquilibrationSettings(),
            )
            logger.info("Using MMGBSA lowest-energy snapshot as protein starting structure")
            bound_state = build_bound_state_from_state(snapshot)

            save_prepared_state(bound_state, equil_dir, "bound")
            save_prepared_state(unbound_state, equil_dir, "unbound")
            scores_path.write_text(json.dumps(mmgbsa_scores))

        logger.info("Collecting escape trajectories (unbound first, then bound replicas)")
        data = collect_trajectories(
            bound_state,
            unbound_state,
            config,
            checkpoint_dir=checkpoint_dir,
            trajectory_dir=trajectory_dir,
            resume=True,
        )

        logger.info("Analyzing")
        results = analyze_modbinddg(data, config, work_dir)
        results["mmgbsa_score"] = mmgbsa_scores.get("mmgbsa_score", float("nan"))

        # results.csv is written into work_dir by analyze_modbinddg. For a local
        # output that already IS the destination; for a remote output, publish it.
        if remote:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "results.csv").write_bytes((work_dir / "results.csv").read_bytes())
            logger.info(f"Published results to {out_dir / 'results.csv'}")
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
@click.option(
    "--temperature",
    type=float,
    default=900.0,
    show_default=True,
    help="Bound-state (unbinding) simulation temperature (K). The unbound "
    "ligand-in-solvent state is always run at 300 K (the reweighting reference).",
)
@click.option("--n-replicas", default=8, show_default=True, help="Number of bound escape replicas.")
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
    temperature: float,
    n_replicas: int,
    platform: str | None,
) -> None:
    """Run ModBinddG from the command line."""
    set_platform(platform)
    config = ModBindDGSettings(
        bound_temperature=temperature * unit.kelvin,
        unbound_temperature=temperature * unit.kelvin,
        n_replicas=n_replicas,
        bound_box_shape="dodecahedron",  # type: ignore[arg-type]
    )
    results = run_modbind(protein, ligand, output, config=config)
    logger.info(f"Done: ΔG° = {results['delta_g']:.2f} kcal/mol")
    logger.info(
        f"Total simulation time = {results['total_sim_time_ns']:.2f} ns "
        f"(bound {results['bound_sim_time_ns']:.2f} ns, "
        f"unbound {results['unbound_sim_time_ns']:.2f} ns)"
    )


if __name__ == "__main__":
    main()
