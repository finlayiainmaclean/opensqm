"""SQM module."""

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Final

import click
import pandas as pd
import ray
import scipy
import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from sqlitedict import SqliteDict
from tqdm.auto import tqdm
from unipka import UnipKa

from opensqm.md.relax import relax_complex
from opensqm.mopac import (
    optimise_complex,
    run_interaction_energy,
    run_opt_from_rdmol,
    run_singlepoint_from_rdmol,
)
from opensqm.rdkit_utils import crop_and_cap_protein, set_residue_info

logger.remove()
logger.add(sys.stderr, level="INFO")

NUM_CPUS: Final[int] = int(os.environ.get("NUM_CPUS", "5"))


class OptimisationSettings(BaseModel):
    """Settings for the optimisation process."""

    model_config = ConfigDict(frozen=True)

    sqm_optimise: bool = False
    mm_optimise: bool = False
    crop: bool = True


class Complex(BaseModel):
    """Paths to protein PDB and ligand MOL for one SQM run."""

    model_config = ConfigDict(frozen=True)
    complex_id: str
    protein_file: Path
    ligand_file: Path


class SQMConfig(BaseModel):
    """Paths to protein PDB and ligand MOL for one SQM run."""

    model_config = ConfigDict(frozen=True)
    complex: Complex
    settings: OptimisationSettings


class SQMOutput(BaseModel):
    """Interaction energies and combined score from one SQM run."""

    model_config = ConfigDict(frozen=True)

    interaction_energy: float
    ligand_strain: float
    E_complex: float
    E_protein: float
    E_ligand: float
    G_Hplus: float
    score: float


@ray.remote
def run_sqm_wrapper(inp: SQMConfig, output_dir: Path) -> SQMOutput | None:
    """Run SQM on a protein and ligand with caching."""
    config_hash = hashlib.sha256(inp.json(sort_keys=True).encode("utf-8")).hexdigest()[:16]
    output_json = output_dir / f"{config_hash}.json"

    if output_json.exists():
        logger.info(f"Output existing for {inp.complex.complex_id}")
        return SQMOutput.parse_raw(output_json.read_text())

    logger.info(f"Running SQM for {inp.complex.complex_id}")
    try:
        output = run_sqm(inp)
        output_json.write_text(output.json(sort_keys=True))
        return output
    except Exception as e:
        logger.error(f"Error running SQM for {inp.complex.complex_id}: {e}")
        return None


def run_sqm(inp: SQMConfig) -> SQMOutput:
    """Run SQM on protein and ligand."""
    logger.info(
        f"Processing Protein: {inp.complex.protein_file} | Ligand: {inp.complex.ligand_file}"
    )
    ligand = Chem.MolFromMolFile(str(inp.complex.ligand_file), removeHs=False)

    protein = Chem.MolFromPDBFile(
        str(inp.complex.protein_file), removeHs=False, proximityBonding=True, sanitize=False
    )

    unipka = UnipKa()
    ligand_no_h = Chem.RemoveHs(ligand)
    dist_df = unipka.get_distribution(ligand_no_h, pH=7.4)
    G_Hplus = dist_df[dist_df.is_query_mol].relative_ph_adjusted_free_energy.iloc[0]
    G_Hplus = float(G_Hplus)

    ligand = set_residue_info(ligand)

    Chem.rdmolops.Kekulize(protein)
    Chem.rdmolops.Kekulize(ligand)

    if inp.settings.crop:
        logger.info("Cropping protein")
        protein = crop_and_cap_protein(protein=protein, ligand=ligand, distance_to_ligand=11)

    if inp.settings.mm_optimise:
        logger.info("Minimising complex")
        ligand, protein = relax_complex(ligand=ligand, protein=protein, simulation_time=2)

    if inp.settings.sqm_optimise:
        logger.info("Crudely optimising ligand")

        # ligand, protein = optimise_complex(
        #     ligand=ligand,
        #     protein=protein,
        #     mode="ligand",
        #     gnorm=20,
        #     num_epochs=3,  # ~4 minutes an epoch
        #     use_rapid=True,
        # )

        logger.info("Refining ligand")
        ligand, protein = optimise_complex(
            ligand=ligand,
            protein=protein,
            mode="hydrogens",
            gnorm=10,
            num_epochs=3,  # # ~5 minutes an epoch
            use_rapid=False,
        )

        # for epoch in tqdm(range(20)):
        #     ligand, protein = optimise_complex(
        #         ligand=ligand,
        #         protein=protein,
        #         mode="ligand",
        #         gnorm=20,
        #         num_epochs=1,  # ~4 minutes an epoch
        #         use_rapid=True,
        #     )

        # for epoch in tqdm(range(3)):
        #     logger.info(f"Refining ligand")
        #     ligand, protein = optimise_complex(
        #         ligand=ligand,
        #         protein=protein,
        #         mode="ligand",
        #         gnorm=10,
        #         num_epochs=1,  # # ~5 minutes an epoch
        #         use_rapid=False,
        # )

    ligand_charge = Chem.GetFormalCharge(ligand)
    E_ligand_bound = run_singlepoint_from_rdmol(
        ligand, use_mozyme=True, solvent="cosmo2", charge=ligand_charge
    )

    ligand_free = run_opt_from_rdmol(
        ligand, mopac_keywords=["PM6-D3H4X", "EPS=78.5"], charge=ligand_charge
    )

    ligand_rmsd = rdMolAlign.CalcRMS(ligand_free, ligand)

    logger.info(f"Ligand RMSD: {ligand_rmsd}")

    E_ligand_free = run_singlepoint_from_rdmol(
        ligand_free, use_mozyme=True, solvent="cosmo2", charge=ligand_charge
    )

    ligand_strain = E_ligand_bound - E_ligand_free

    scores = run_interaction_energy(ligand=ligand, protein=protein)
    score = scores["interaction_energy"] + G_Hplus  # ligand_strain

    logger.info(
        f"interaction_energy: {scores['interaction_energy']:.2f}, G_Hplus: {G_Hplus:.2f}, "
        f"ligand_strain: {ligand_strain:.2f}, score: {score:.2f}"
    )

    return SQMOutput(
        interaction_energy=scores["interaction_energy"],
        E_complex=scores["E_complex"],
        E_protein=scores["E_protein"],
        E_ligand=scores["E_ligand"],
        G_Hplus=G_Hplus,
        ligand_strain=ligand_strain,
        score=score,
    )


def process_ray_futures(futures: list[ray.ObjectRef], desc: str = "Processing") -> list[Any]:
    """Process ray futures with a tqdm progress bar, returning results in original order."""
    results: dict[int, Any] = {}
    future_to_idx = {f: i for i, f in enumerate(futures)}
    unready = list(futures)

    with tqdm(total=len(futures), desc=desc) as pbar:
        while unready:
            ready, unready = ray.wait(unready, num_returns=1)
            for f in ready:
                results[future_to_idx[f]] = ray.get(f)
                pbar.update(1)

    return [results[i] for i in range(len(futures))]


def _run_local_env() -> bool:
    v = os.environ.get("RUN_LOCAL", "").strip()
    return v == "1" or v.lower() == "true"


@click.command()
@click.argument("dataframe-path", type=click.Path(exists=True, path_type=Path))
@click.argument("settings-path", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="data/outputs",
    help="Directory to save outputs",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing results in the database",
)
def cli(
    dataframe_path: Path,
    settings_path: Path | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> None:
    """Run the SQM CLI."""
    output_dir = output_dir or Path("data/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)

    if settings_path is not None:
        with Path(settings_path).open() as f:
            settings_dict = yaml.safe_load(f)
        settings = OptimisationSettings(**settings_dict)
    else:
        settings = OptimisationSettings()

    run_local = _run_local_env()
    if not run_local:
        ray.init(num_cpus=NUM_CPUS, ignore_reinit_error=True)

    df = pd.read_csv(dataframe_path)
    if "protein" in df.columns:
        df = df.dropna(subset=["protein"])

    db = SqliteDict("data/outputs/results.sqlite", autocommit=True)

    print(len(db))

    inputs_to_run = []
    all_results = {}

    for row in df.itertuples():
        complex_id = str(row.id)  # type: ignore
        if not overwrite and complex_id in db:
            result_json = db[complex_id]
            all_results[complex_id] = SQMOutput.parse_raw(result_json)
            continue

        inputs_to_run.append(
            SQMConfig(
                complex=Complex(
                    complex_id=complex_id,
                    protein_file=Path(row.protein),  # type: ignore
                    ligand_file=Path(row.ligand),  # type: ignore
                ),
                settings=settings,
            )
        )

    if inputs_to_run:
        if run_local:
            new_results = [run_sqm(inp) for inp in tqdm(inputs_to_run, desc="SQM")]
        else:
            futures = [run_sqm_wrapper.remote(inp, output_dir) for inp in inputs_to_run]
            new_results = process_ray_futures(futures, desc="Processing")
            new_results = [r for r in new_results if r is not None]

        for inp, result in zip(inputs_to_run, new_results, strict=False):
            all_results[inp.complex.complex_id] = result
            db[inp.complex.complex_id] = result.json(sort_keys=True)

    db.close()

    results_list = [
        all_results[str(row.id)] for row in df.itertuples() if str(row.id) in all_results
    ]

    scores_df = pd.DataFrame([r.dict() for r in results_list])
    scores_df["complex_id"] = [str(row.id) for row in df.itertuples() if str(row.id) in all_results]
    if len(scores_df) > 1 and "pX" in df.columns:
        scores_df = scores_df.loc[scores_df.groupby("complex_id")["interaction_energy"].idxmin()]
        corr = scipy.stats.spearmanr(scores_df["interaction_energy"], df.loc[scores_df.index, "pX"])
        logger.info(f"Spearman correlation: {corr}")

    scores_csv_path = output_dir / "scores.csv"
    scores_df.to_csv(scores_csv_path, index=False)
    logger.info(f"Saved scores to {scores_csv_path}")

    if "pX" in df.columns:
        corr = scipy.stats.spearmanr(scores_df["interaction_energy"], df.loc[scores_df.index, "pX"])
        logger.info(f"Spearman correlation: {corr}")


if __name__ == "__main__":
    cli()
