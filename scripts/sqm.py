"""SQM module."""

import os
import sys
from pathlib import Path

import click
import pandas as pd
import ray
import scipy
import xxhash
import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict
from rdkit import Chem
from rdkit.Chem import rdMolAlign
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


class OptimisationSettings(BaseModel):
    """Settings for the optimisation process."""

    sqm_optimise: bool = True
    mm_optimise: bool = True
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

    def __hash__(self) -> int:
        """xxHash-64 of complex id and both paths (POSIX), in order."""
        h = xxhash.xxh64()
        h.update(self.complex.complex_id.encode("utf-8"))
        h.update(b"\0")
        for path in (self.complex.protein_file, self.complex.ligand_file):
            h.update(path.expanduser().as_posix().encode("utf-8"))
            h.update(b"\0")
        return h.intdigest()


class SQMOutput(BaseModel):
    """Interaction energies and combined score from one SQM run."""

    model_config = ConfigDict(frozen=True)

    dE_int: float
    dE_ligand_strain: float
    E_complex: float
    E_protein: float
    E_ligand: float
    G_Hplus: float
    score: float


@ray.remote
def run_sqm_wrapper(inp: SQMConfig, output_dir: Path) -> SQMOutput:
    """Run SQM on a protein and ligand with caching."""
    output_json = output_dir / f"{hash(inp)}.json"

    if output_json.exists():
        logger.info(f"Output existing for {inp.complex.complex_id}")
        return SQMOutput.model_validate_json(output_json.read_text())

    logger.info(f"Running SQM for {inp.complex.complex_id}")
    output = run_sqm(inp)
    output_json.write_text(output.model_dump_json())
    return output


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
        protein = crop_and_cap_protein(protein=protein, ligand=ligand, distance_to_ligand=11)

    if inp.settings.mm_optimise:
        ligand, protein = relax_complex(ligand=ligand, protein=protein, simulation_time=60)

    if inp.settings.sqm_optimise:
        ligand, protein = optimise_complex(
            ligand=ligand,
            protein=protein,
            mode="ligand",
            gnorm=20,
            num_epochs=5,  # ~4 minutes an epoch
            use_rapid=True,
        )
        ligand, protein = optimise_complex(
            ligand=ligand,
            protein=protein,
            mode="ligand",
            gnorm=10,
            num_epochs=3,  # # ~5 minutes an epoch
            use_rapid=False,
        )

    ligand_charge = Chem.GetFormalCharge(ligand)
    E_ligand_bound = run_singlepoint_from_rdmol(
        ligand, use_mozyme=True, solvent="cosmo2", charge=ligand_charge
    )

    ligand_free = run_opt_from_rdmol(
        ligand, mopac_keywords=["PM6-D3H4X", "EPS=78.5"], charge=ligand_charge
    )

    try:
        ligand_rmsd = rdMolAlign.CalcRMS(ligand_free, ligand)
    except Exception as e:
        Chem.MolToMolFile(ligand_free, "/tmp/ligand_free.sdf")
        Chem.MolToMolFile(ligand, "/tmp/ligand.sdf")
        raise ValueError("Could not calculate ligand RMSD") from e

    logger.info(f"Ligand RMSD: {ligand_rmsd}")

    E_ligand_free = run_singlepoint_from_rdmol(
        ligand_free, use_mozyme=True, solvent="cosmo2", charge=ligand_charge
    )

    dE_ligand_strain = E_ligand_bound - E_ligand_free

    scores = run_interaction_energy(ligand=ligand, protein=protein)
    score = scores["dE_int"] + G_Hplus + dE_ligand_strain

    logger.info(
        f"dE_int: {scores['dE_int']:.2f}, G_Hplus: {G_Hplus:.2f}, "
        f"dE_ligand_strain: {dE_ligand_strain:.2f}, score: {score:.2f}"
    )

    return SQMOutput(
        dE_int=scores["dE_int"],
        E_complex=scores["E_complex"],
        E_protein=scores["E_protein"],
        E_ligand=scores["E_ligand"],
        G_Hplus=G_Hplus,
        dE_ligand_strain=dE_ligand_strain,
        score=score,
    )


def _run_local_env() -> bool:
    v = os.environ.get("RUN_LOCAL", "").strip()
    return v == "1" or v.lower() == "true"


@click.command()
@click.argument("dataframe_path", type=click.Path(exists=True, path_type=Path))
@click.argument("settings_path", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="data/outputs",
    help="Directory to save outputs",
)
def cli(dataframe_path: Path, settings_path: Path | None = None, output_dir: Path | None = None):
    """Run the SQM CLI."""
    output_dir = output_dir or Path("data/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)

    if settings_path is not None:
        with open(settings_path) as f:
            settings_dict = yaml.safe_load(f)
        settings = OptimisationSettings(**settings_dict)
    else:
        settings = OptimisationSettings()

    run_local = _run_local_env()
    if not run_local:
        ray.init(num_cpus=5, ignore_reinit_error=True)

    df = pd.read_csv(dataframe_path)
    if "protein" in df.columns:
        df = df.dropna(subset=["protein"])

    inputs = [
        SQMConfig(
            complex=Complex(
                complex_id=str(row.id),
                protein_file=Path(row.protein),
                ligand_file=Path(row.ligand),
            ),
            settings=settings,
        )
        for row in df.itertuples()
    ]

    if run_local:
        results = [run_sqm(inp) for inp in tqdm(inputs, desc="SQM")]
    else:
        futures = [
            run_sqm_wrapper.remote(inp, output_dir)
            for inp in tqdm(inputs, total=len(inputs), desc="Submitting")
        ]
        results = ray.get(futures)

    scores_df = pd.DataFrame([r.model_dump() for r in results])
    if len(scores_df) > 1 and "pX" in df.columns:
        corr = scipy.stats.spearmanr(scores_df["score"], df["pX"])
        logger.info(f"Spearman correlation: {corr}")

    scores_csv_path = output_dir / "scores.csv"
    scores_df.to_csv(scores_csv_path, index=False)
    logger.info(f"Saved scores to {scores_csv_path}")


if __name__ == "__main__":
    cli()
