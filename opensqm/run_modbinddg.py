"""ModBinddG: absolute binding free energy via population reweighting.

Command-line entry point for the two-state population-reweighting method of
Sinko et al. (*PNAS* 2026). Builds the bound (ligand-protein) and unbound
(ligand-in-solvent) states, runs high-temperature escape simulations, and
reports ``ΔG°`` with a bootstrapped confidence interval.
"""

import json
from pathlib import Path

import click
from loguru import logger
from openmm import unit
from opensqm.fix import run_pdbfixer
from opensqm.md.equilibrate import EquilibrationSettings
from rdkit import Chem, RDLogger

from opensqm.modbinddg.analyze import analyze_modbinddg
from opensqm.modbinddg.config import ModBindDGSettings
from opensqm.modbinddg.simulate import collect_trajectories
from opensqm.modbinddg.states import (
    build_bound_state_from_state,
    build_unbound_state,
    load_prepared_state,
    save_prepared_state,
)
from opensqm.run_cph import ConstantpHRunSettings, pHResult, run_cph

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


def _write_manifest(output_path: Path, *, protein: str, ligand: str, config: ModBindDGSettings) -> None:
    payload = {
        "version": 1,
        "protein": str(Path(protein).resolve()),
        "ligand": str(Path(ligand).resolve()),
        "config_hash": config.hash(),
        "n_replicas": config.n_replicas,
        "temperature_K": list(config.bound_temperatures_K()),
    }
    (output_path / MANIFEST_FILENAME).write_text(json.dumps(payload, indent=2))


def _check_manifest(output_path: Path, *, protein: str, ligand: str, config: ModBindDGSettings) -> None:
    path = output_path / MANIFEST_FILENAME
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    if manifest.get("config_hash") != config.hash():
        raise ValueError(
            "Existing run manifest has a different config hash; refusing to "
            "resume with mismatched settings. Use a fresh output directory."
        )


def run_modbinddg(
    protein: str,
    ligand: str,
    output: str,
    resume: bool = True,
    config: ModBindDGSettings = ModBindDGSettings(),
    
) -> dict:
    """Run a full ModBinddG calculation for one protein-ligand pair.

    Constant-pH MD at pH 7 is used as an equilibration step to determine the
    dominant protonation state and provide an equilibrated protein structure.
    The lowest-energy snapshot from that 1 ns run is then used as the starting
    conformation for the escape simulations.
    """
    output_path = Path(output)
    protein = Path(protein)
    ligand = Path(ligand)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / "checkpoints"
    fixed_protein = output_path / "protein_prepared.pdb"
    states_dir = output_path / "equilibrated_states"

    run_pdbfixer(protein, fixed_protein)

    config.cph_equilibration_ns = 0.1

    # --- constant-pH equilibration at pH 7 ---
    logger.info(f"Running {config.cph_equilibration_ns} ns CpH equilibration at pH 7")
    cph_result = run_cph(
        fixed_protein,
        output=str(output_path / "cph_equilibration"),
        ligand=ligand,
        config=ConstantpHRunSettings(
            pH=7.0,
            production_time=config.cph_equilibration_ns * unit.nanosecond,
            use_ph_remd=False,
            n_replicas=1,
            titratable_residue_query="(protein within 5 of resn LIG) or (resn LIG)",
        ),
        resume=resume,
    )
    cph_result: pHResult = cph_result["ph_results"][0] # ph 7
    snapshot = cph_result.lowest_energy_snapshot
    logger.info(f"Lowest-energy snapshot at pH 7: {cph_result.population}")

    need_unbound = config.unbound_mode == "explicit"
    bound_state = load_prepared_state(states_dir, "bound") if resume else None
    unbound_state = (
        load_prepared_state(states_dir, "unbound") if (resume and need_unbound) else None
    )
    if bound_state is not None:
        logger.info("Loaded equilibrated bound state from cache")
    if need_unbound and unbound_state is not None:
        logger.info("Loaded equilibrated unbound state from cache")

    if bound_state is None or (need_unbound and unbound_state is None):
        from unipka import UnipKa
        unipka = UnipKa()

        ligand_rdmol = Chem.MolFromMolFile(ligand, removeHs=True)
        df = unipka.get_distribution(ligand_rdmol)
        ligand_rdmol = df.iloc[0].mol
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)

        if need_unbound and unbound_state is None:
            logger.info("Building and equilibrating unbound state")
            unbound_state = build_unbound_state(
                ligand_rdmol,
                equilibration_config=EquilibrationSettings(),
                bespoke_ligand_forcefield=config.bespoke_ligand_forcefield,
            )
            save_prepared_state(unbound_state, states_dir, "unbound")

        if bound_state is None:
           
            logger.info("Using CpH lowest-energy snapshot as protein starting structure")
            bound_state = build_bound_state_from_state(
                snapshot,
                ligand_rdmol=ligand_rdmol,
                bespoke_ligand_forcefield=config.bespoke_ligand_forcefield,
            )
            save_prepared_state(bound_state, states_dir, "bound")

    trajectory_dir = output_path / "trajectories"

    logger.info("Collecting escape trajectories (unbound first, then bound replicas)")
    data = collect_trajectories(
        bound_state,
        unbound_state,
        config,
        checkpoint_dir=checkpoint_dir,
        trajectory_dir=trajectory_dir,
        resume=resume,
    )

    if data.bound_temperatures_K:
        config = config.copy(
            update={
                "temperature": data.bound_temperatures_K[0] * unit.kelvin,
                "replica_temperatures": tuple(data.bound_temperatures_K),
            }
        )

    _write_manifest(output_path, protein=protein, ligand=ligand, config=config)

    logger.info("Analyzing")
    results = analyze_modbinddg(
        data,
        config,
        output_path,
        ligand_path=ligand,
        trajectory_dir=trajectory_dir,
    )
    return results


@click.command()
@click.option("--protein", required=True, type=click.Path(exists=True), help="Protein PDB file.")
@click.option("--ligand", required=True, type=click.Path(exists=True), help="Ligand MOL/SDF file.")
@click.option("--output", required=True, type=click.Path(), help="Output directory.")
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
@click.option(
    "--bound-box-shape",
    type=click.Choice(["cube", "dodecahedron", "octahedron"]),
    default="dodecahedron",
    show_default=True,
    help="Periodic box shape for the bound complex.",
)
@click.option(
    "--n-closest-waters",
    default=5,
    show_default=True,
    help="Waters kept per frame for bound-state MMGBSA.",
)
def main(
    protein: str,
    ligand: str,
    output: str,
    temperature: str,
    ideal_escape_time: float,
    n_replicas: int,
    unbound_mode: str,
    bound_box_shape: str,
    n_closest_waters: int,
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
        bound_box_shape=bound_box_shape,  # type: ignore[arg-type]
        n_closest_waters=n_closest_waters,
    )
    results = run_modbinddg(protein, ligand, output, config=config, resume=True)
    logger.info(f"Done: ΔG° = {results['delta_g']:.2f} kcal/mol")
    logger.info(
        f"Bound MMGBSA = {results['mmgbsa_mean']:.2f} "
        f"+/- {results['mmgbsa_std']:.2f} kcal/mol"
    )
    logger.info(
        f"Total simulation time = {results['total_sim_time_ns']:.2f} ns "
        f"(bound {results['bound_sim_time_ns']:.2f} ns, "
        f"unbound {results['unbound_sim_time_ns']:.2f} ns)"
    )


if __name__ == "__main__":
    main()
