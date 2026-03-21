"""Download and preprocess the PL-REX dataset."""

from pathlib import Path
from typing import Final

import pandas as pd
import wget
from rdkit import Chem
from tqdm import tqdm

from opensqm.fix import run_pdbfixer

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

            raw_prot_pdb_path = target_dir / f"{pdb_code}.raw.pdb"
            fixed_prot_pdb_path = target_dir / f"{pdb_code}.fixed.pdb"

            PDB_CODE_LENGTH: Final[int] = 4
            if len(pdb_code) == PDB_CODE_LENGTH:
                raw_prot_pdb_path = target_dir / f"{pdb_code}.raw.pdb"
                fixed_prot_pdb_path = target_dir / f"{pdb_code}.fixed.pdb"

                if not fixed_prot_pdb_path.exists():
                    print(url := f"https://files.rcsb.org/download/{pdb_code}.pdb")
                    wget.download(
                        url,
                        out=str(raw_prot_pdb_path),
                    )
                    run_pdbfixer(raw_prot_pdb_path, fixed_prot_pdb_path, keep_waters=True)
            else:
                fixed_prot_pdb_path = None

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
                    "rscb_protein": fixed_prot_pdb_path,
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

        _inputs = inputs.drop(columns=["protein"]).rename(columns={"rscb_protein": "protein"})
        file = target_dir / "rscb_protein.csv"
        _inputs.to_csv(file)  # Config for RCSB protein

        _inputs = inputs.drop(columns=["protein"]).rename(columns={"pocket": "protein"})
        file = target_dir / "pocket.csv"
        _inputs.to_csv(file)  # Config for precropped pocket

        file = target_dir / "protein.csv"
        _inputs = inputs.drop(columns=["pocket"])
        _inputs.to_csv(file)  # Config for full protein


if __name__ == "__main__":
    main()
