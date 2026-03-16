"""Download and preprocess the PL-REX dataset."""

from pathlib import Path
from typing import Final

import pandas as pd
import wget
from loguru import logger
from rdkit import Chem
from tqdm import tqdm

DATASET = "PL-REX"
BASE_URL = "https://raw.githubusercontent.com/Honza-R/PL-REX/refs/heads/main"

TARGET_NAMES: Final[list[str]] = [
    "001-CA2",
    "002-HIV-PR",
    "003-CK2",
    "004-AR",
    "005-Cath-D",
    "006-BACE1",
    "007-JAK1",
    "008-Trypsin",
    "009-CDK2",
    "010-MMP12",
]


def main():
    """Download PL-REX dataset."""
    for target_name in tqdm(TARGET_NAMES):
        _df = pd.read_csv(
            f"https://raw.githubusercontent.com/Honza-R/PL-REX/refs/heads/main/{target_name}/experimental_dG.txt",
            skiprows=2,
            sep=r"\s+",
            header=None,
        )

        _df.columns = ["pdb_code", "pX"]
        _df["pX"] = -_df["pX"]
        _df["target_name"] = target_name

        target_dir = Path("data") / "inputs" / DATASET / target_name
        target_dir.mkdir(exist_ok=True, parents=True)

        inputs = []

        for row in _df.to_dict("records"):
            pdb_code = row["pdb_code"]
            pX = row["pX"]
            prot_pdb_path = target_dir / f"{pdb_code}.prot.pdb"

            pocket_pdb_path = target_dir / f"{pdb_code}.pocket.pdb"

            lig_sdf_path = target_dir / f"{pdb_code}.sdf"

            if not pocket_pdb_path.exists():
                wget.download(
                    f"{BASE_URL}/{target_name}/structures_pl-rex/{pdb_code}/receptor.pdb",
                    out=str(pocket_pdb_path),
                )

            if not prot_pdb_path.exists():
                wget.download(
                    f"{BASE_URL}/{target_name}/structures_pl-rex/{pdb_code}/protein.pdb",
                    out=str(prot_pdb_path),
                )

            if not lig_sdf_path.exists():
                wget.download(
                    f"{BASE_URL}/{target_name}/structures_pl-rex/{pdb_code}/ligand.sdf",
                    out=str(lig_sdf_path),
                )

            ligand = Chem.MolFromMolFile(str(lig_sdf_path), removeHs=False)

            smi = Chem.MolToSmiles(Chem.RemoveHs(ligand))

            inchikey = Chem.MolToInchiKey(ligand)

            inputs.append(
                {
                    "protein": prot_pdb_path,
                    "pocket": pocket_pdb_path,
                    "ligand": lig_sdf_path,
                    "target_name": target_name,
                    "inchikey": inchikey,
                    "id": pdb_code,
                    "smi": smi,
                    "pX": pX,
                }
            )

        inputs = pd.DataFrame(inputs)

        _inputs = inputs.drop(columns=["protein"]).rename(columns={"pocket": "protein"})
        file = target_dir / "pocket.csv"

        logger.info(f"Saving to {file!s}")

        _inputs.to_csv(file)  # Config for precropped pocket

        file = target_dir / "protein.csv"
        logger.info(f"Saving to {file!s}")

        _inputs = inputs.drop(columns=["pocket"])

        _inputs.to_csv(file)  # Config for full protein


if __name__ == "__main__":
    main()
