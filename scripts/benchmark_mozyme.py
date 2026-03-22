"""SQM module."""

import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from rdkit import Chem
from tqdm.auto import tqdm

from opensqm.mopac import get_correct_ligand
from opensqm.rdkit_utils import set_residue_info

project_dir = Path("data/inputs/PL-REX/")


logger.remove()
logger.add(sys.stderr, level="INFO")


def _run(*, ligand_file: Path):
    """Run SQM on protein and ligand."""
    ligand = Chem.MolFromMolFile(str(ligand_file), removeHs=False)

    ligand = set_residue_info(ligand)

    Chem.rdmolops.Kekulize(ligand)

    _best_ligand, best_dE = get_correct_ligand(ligand)

    return best_dE


if __name__ == "__main__":
    import numpy as np

    energies = []

    targets_dirs = project_dir.glob("*")

    for target_dir in tqdm(list(targets_dirs)):
        df = pd.read_csv(target_dir / "pocket.csv")

        for _i, row in enumerate(df.itertuples()):
            energy = _run(ligand_file=row.ligand)  # type: ignore
            energies.append(energy)

    print(np.mean(energies))
    print(np.max(energies))
