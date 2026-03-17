"""SQM module."""

from opensqm.mopac import get_correct_ligand
import sys
from pathlib import Path

import pandas as pd
import scipy
from loguru import logger
from rdkit import Chem
from tqdm.auto import tqdm
from unipka import UnipKa

from opensqm._types import Complex
from opensqm.mopac import (
    fix_nitro_groups,
    run_interaction_energy,
)
from opensqm.rdkit_utils import set_residue_info

project_dir = Path("data/inputs/PL-REX/")



logger.remove()
logger.add(sys.stderr, level="INFO")


def _run(*, ligand_file: Path):
    """Run SQM on protein and ligand."""
    ligand = Chem.MolFromMolFile(str(ligand_file), removeHs=False)

    ligand = set_residue_info(ligand)
    ligand = fix_nitro_groups(ligand)

    Chem.rdmolops.Kekulize(ligand)

    best_ligand, best_dE = get_correct_ligand(ligand)

    return best_dE

    


if __name__ == "__main__":
    import numpy as np
    dEs = []

    targets_dirs = project_dir.glob("*")

    for target_dir in tqdm(list(targets_dirs)):
        df = pd.read_csv(target_dir / "pocket.csv")

        for i, row in enumerate(df.itertuples()):
            dE = _run(ligand_file=row.ligand)  # type: ignore
            dEs.append(dE)

    print(np.mean(dEs))
    print(np.max(dEs))
    
