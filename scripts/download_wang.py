"""Download and preprocess the PL-REX dataset."""

from pathlib import Path
from typing import Final

import pandas as pd
import wget
from rdkit import Chem
from tqdm import tqdm

from opensqm.fix import run_pdbfixer

DATASET = "Wang"

BASE_URL = "https://raw.githubusercontent.com/Honza-R/Wang_dataset_SQM/refs/heads/main/Zariquiey_structures_extended"

TARGET_NAMES: Final[list[str]] = ["BACE", "CDK2", "JNK1", "MCL1", "PTP1", "Tyk2", "p38", "thrombin"]

TARGET_PDB_CODES: Final[dict[str, str]] = {
    "BACE": "4DJX",
    "CDK2": "1H1Q",
    "JNK1": "2GMX",
    "MCL1": "4HW3",
    "PTP1": "2QBS",
    "Tyk2": "4GIH",
    "p38": "3FLY",
    "thrombin": "2ZFF",
}


def main():
    """Download PL-REX dataset."""
    for target_name in tqdm(TARGET_NAMES):
        # Each molecule has several candidate poses; SQM2.20_selected_poses.txt
        # names the single pose SQM2.20 selected for each (its directory under
        # structures/ is named exactly by that pose id).
        poses_url = f"{BASE_URL}/{target_name}/SQM2.20_selected_poses.txt"
        print(poses_url)
        poses_df = pd.read_csv(
            poses_url,
            sep=r"\s+",
            header=None,
            names=["pose_id", "sqm_score"],
        )

        # Experimental ΔG is keyed per pose id (the base molecule and each of its
        # poses share the same value, e.g. ejm_31 and ejm_31_pose_orig).
        dg_url = f"{BASE_URL}/{target_name}/experimental_dG.txt"
        dg_df = pd.read_csv(dg_url, sep=r"\s+", header=None, names=["id", "dG"])
        dG_by_id = dict(zip(dg_df["id"], dg_df["dG"], strict=False))

        target_dir = Path("data") / "inputs" / DATASET / target_name
        target_dir.mkdir(exist_ok=True, parents=True)

        inputs = []

        for row in poses_df.to_dict("records"):
            pdb_code = TARGET_PDB_CODES[target_name]
            pose_id = row["pose_id"]
            # Strip the "_pose_*" suffix to recover the base molecule id.
            mol_id = pose_id.split("_pose")[0]
            dG = dG_by_id.get(pose_id, dG_by_id.get(mol_id))

            lig_sdf_path = target_dir / f"{mol_id}.sdf"

            raw_prot_pdb_path = target_dir / f"{pdb_code}.raw.pdb"
            fixed_prot_pdb_path = target_dir / f"{pdb_code}.fixed.pdb"

            if not fixed_prot_pdb_path.exists():
                URL = f"{BASE_URL}/{target_name}/structures/{pose_id}/protein.pdb"
                wget.download(
                    URL,
                    out=str(raw_prot_pdb_path),
                )
                run_pdbfixer(raw_prot_pdb_path, fixed_prot_pdb_path, keep_waters=True)

            if not lig_sdf_path.exists():
                URL = f"{BASE_URL}/{target_name}/structures/{pose_id}/ligand.sdf"
                print(URL)
                wget.download(
                    URL,
                    out=str(lig_sdf_path),
                )

            ligand = Chem.MolFromMolFile(str(lig_sdf_path), removeHs=False)

            smi = Chem.MolToSmiles(Chem.RemoveHs(ligand))

            inchikey = Chem.MolToInchiKey(ligand)

            inputs.append(
                {
                    "protein": fixed_prot_pdb_path,
                    "ligand": lig_sdf_path,
                    "target_name": target_name,
                    "inchikey": inchikey,
                    "id": pdb_code,
                    "pose_id": pose_id,
                    "smi": smi,
                    # benchmark_wang.py reads "pX", which stores -ΔG in kcal/mol.
                    "pX": -dG,
                }
            )

        inputs = pd.DataFrame(inputs)

        file = target_dir / "protein.csv"
        inputs.to_csv(file)


if __name__ == "__main__":
    main()
