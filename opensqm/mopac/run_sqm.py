"""Run single-complex SQM calculation."""

import sys
from pathlib import Path

import click
import pandas as pd
import yaml
from loguru import logger

from opensqm.mopac.run_sqm_ray import Complex, OptimisationSettings, SQMConfig, run_sqm

logger.remove()
logger.add(sys.stderr, level="INFO")


@click.command()
@click.option(
    "--protein",
)
@click.option(
    "--ligand",
)
@click.option(
    "--output",
)
@click.option(
    "--settings-path",
    required=False,
)
def main(
    protein: str,
    ligand: str,
    output: str,
    settings_path: str,
) -> None:
    """Run entrypoint for Single-Complex SQM calculation."""
    if output is None:
        output_path = Path("data/outputs")
    else:
        output_path = Path(output)

    output_path.mkdir(exist_ok=True, parents=True)

    protein_path = Path(protein)
    ligand_path = Path(ligand)

    if settings_path is not None:
        with Path(settings_path).open() as f:
            settings_dict = yaml.safe_load(f)
        settings = OptimisationSettings(**settings_dict)
    else:
        settings = OptimisationSettings(mm_optimise=True)

    complex_id = f"{protein_path.stem}_{ligand_path.stem}"

    inp = SQMConfig(
        complex=Complex(
            complex_id=complex_id,
            protein_file=protein_path,
            ligand_file=ligand_path,
        ),
        settings=settings,
    )

    logger.info(f"Running SQM for {complex_id}")
    result = run_sqm(inp)

    score_path = output_path / "scores.csv"

    result_dict = result.model_dump()
    result_dict["protein"] = str(protein_path)
    result_dict["ligand"] = str(ligand_path)

    scores = pd.Series(result_dict)
    scores.to_csv(score_path, header=False)

    logger.info(f"Saved scores to {score_path}")

    print(scores)


if __name__ == "__main__":
    main()
