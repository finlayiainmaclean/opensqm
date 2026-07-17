"""ModBinddG: absolute binding free energy via population reweighting.

Command-line entry point for the two-state population-reweighting method of
Sinko et al. (*PNAS* 2026). Builds the bound (ligand-protein) and unbound
(ligand-in-solvent) states, runs high-temperature escape simulations, and
reports ``ΔG°`` with a bootstrapped confidence interval.
"""

import tempfile
from pathlib import Path

import click
from cloudpathlib import AnyPath
from loguru import logger
from openmm import unit
from rdkit import RDLogger

from opensqm.cph.run_cph import ConstantpHRunSettings, PHResult, run_cph
from opensqm.fix import run_pdbfixer
from opensqm.md.equilibrate import EquilibrationSettings
from opensqm.modbind.analyze import analyze_modbinddg
from opensqm.modbind.config import ModBindDGSettings
from opensqm.modbind.simulate import collect_trajectories
from opensqm.modbind.states import build_bound_state_from_state, build_unbound_state

RDLogger.DisableLog("rdApp.warning")

MANIFEST_FILENAME = "modbinddg_manifest.json"


def _normalize_replica_temperatures(
    temperatures: tuple[float, ...], n_replicas: int, *, adaptive: bool
) -> tuple[float, ...]:
    if not temperatures:
        raise click.BadParameter("At least one --temperature value is required.")
    if adaptive:
        if len(temperatures) != 1:
            raise click.BadParameter(
                "Adaptive escape tuning uses a single --temperature for replica 0."
            )
        return temperatures
    if len(temperatures) == 1:
        return temperatures * n_replicas
    if len(temperatures) != n_replicas:
        raise click.BadParameter(
            f"Expected 1 or {n_replicas} --temperature values, got {len(temperatures)}."
        )
    return temperatures


def run_modbind(
    protein: str,
    ligand: str,
    output: str,
    config: ModBindDGSettings | None = None,
) -> dict:
    """Run a full ModBinddG calculation for one protein-ligand pair.

    Constant-pH MD at pH 7 is used as an equilibration step to determine the
    dominant protonation state and provide an equilibrated protein structure.
    The lowest-energy snapshot from that 1 ns run is then used as the starting
    conformation for the escape simulations.

    ``protein``, ``ligand`` and ``output`` may each be a local path or an
    ``s3://`` URI. All work runs in a temp dir; only ``results.csv`` is published
    to ``output``.
    """
    if config is None:
        config = ModBindDGSettings()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)

        # Stage inputs locally (downloading from S3 when needed). All heavy work
        # runs in the temp dir; only results.csv is published.
        protein_src, ligand_src = AnyPath(protein), AnyPath(ligand)
        local_protein = tmp_dir / f"protein_input{protein_src.suffix or '.pdb'}"
        local_protein.write_bytes(protein_src.read_bytes())
        local_ligand = tmp_dir / f"ligand_input{ligand_src.suffix or '.sdf'}"
        local_ligand.write_bytes(ligand_src.read_bytes())

        checkpoint_dir = tmp_dir / "checkpoints"
        fixed_protein = tmp_dir / "protein_prepared.pdb"
        trajectory_dir = tmp_dir / "trajectories"

        run_pdbfixer(local_protein, fixed_protein)

        config.cph_equilibration_ns = 0.5

        # --- constant-pH equilibration at pH 7 ---
        logger.info(f"Running {config.cph_equilibration_ns} ns CpH equilibration at pH 7")
        cph_result = run_cph(
            fixed_protein,
            output=str(tmp_dir / "cph_equilibration"),
            ligand=local_ligand,
            config=ConstantpHRunSettings(
                ph=7.0,
                production_time=config.cph_equilibration_ns * unit.nanosecond,
                use_ph_remd=False,
                n_replicas=1,
                protonation_penalty=3.0 * unit.kilocalories_per_mole,
                residue_query="(protein within 5 of resn LIG) or (resn LIG)",
            ),
            resume=False,
        )
        cph_result: PHResult = cph_result["ph_results"][0]  # ph 7
        snapshot = cph_result.lowest_energy_snapshot
        logger.info(f"Lowest-energy snapshot at pH 7: {cph_result.population}")

        unbound_state = None
        if config.unbound_mode == "explicit":
            logger.info("Building and equilibrating unbound state")
            unbound_state = build_unbound_state(
                snapshot.ligand,
                equilibration_config=EquilibrationSettings(),
            )

        logger.info("Using CpH lowest-energy snapshot as protein starting structure")
        bound_state = build_bound_state_from_state(snapshot)

        logger.info("Collecting escape trajectories (unbound first, then bound replicas)")
        data = collect_trajectories(
            bound_state,
            unbound_state,
            config,
            checkpoint_dir=checkpoint_dir,
            trajectory_dir=trajectory_dir,
            resume=False,
        )

        if data.bound_temperatures_k:
            config = config.copy(
                update={
                    "temperature": data.bound_temperatures_k[0] * unit.kelvin,
                    "replica_temperatures": tuple(data.bound_temperatures_k),
                }
            )

        logger.info("Analyzing")
        results = analyze_modbinddg(
            data,
            config,
            tmp_dir,
            ligand_path=local_ligand,
            trajectory_dir=trajectory_dir,
        )

        # Publish only the results CSV to the destination (local dir or S3 prefix).
        out_dir = AnyPath(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_results = out_dir / "results.csv"
        out_results.write_bytes((tmp_dir / "results.csv").read_bytes())
        logger.info(f"Saved results to {out_results}")

    return results


@click.command()
@click.option("--protein", required=True, help="Protein PDB file (local path or s3:// URI).")
@click.option("--ligand", required=True, help="Ligand MOL/SDF file (local path or s3:// URI).")
@click.option("--output", required=True, help="Output directory (local path or s3:// prefix).")
@click.option(
    "--temperature",
    type=str,
    default="900",
    show_default=True,
    help=(
        "Bound-state temperature (K) for each replica. Pass one value to use "
        "the same temperature for all replicas, or one value per replica."
    ),
)
@click.option("--n-replicas", default=8, show_default=True, help="Number of bound escape replicas.")
@click.option(
    "--ideal-escape-time",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "Target bound escape time (ns) for adaptive replica temperatures. "
        "Replica 0 uses --temperature; later replicas are tuned from the "
        "running ΔG°_well estimate. Pass 0 to disable and use a fixed "
        "temperature for every replica."
    ),
)
@click.option(
    "--unbound-mode",
    type=click.Choice(["explicit", "einstein"]),
    default="einstein",
    show_default=True,
    help="Compute the unbound state from explicit MD or the Einstein-Smoluchowski estimate.",
)
def main(
    protein: str,
    ligand: str,
    output: str,
    temperature: str,
    ideal_escape_time: float,
    n_replicas: int,
    unbound_mode: str,
) -> None:
    """Run ModBinddG from the command line."""
    adaptive = ideal_escape_time > 0
    temperature = [float(t) for t in temperature.split(",")]
    ideal_escape_time_ns = ideal_escape_time if adaptive else None
    replica_temperatures = _normalize_replica_temperatures(
        temperature, n_replicas, adaptive=adaptive
    )
    config = ModBindDGSettings(
        temperature=replica_temperatures[0] * unit.kelvin,
        replica_temperatures=replica_temperatures
        if not adaptive and len(set(replica_temperatures)) > 1
        else None,
        ideal_escape_time_ns=ideal_escape_time_ns,
        n_replicas=n_replicas,
        unbound_mode=unbound_mode,  # type: ignore[arg-type]
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
