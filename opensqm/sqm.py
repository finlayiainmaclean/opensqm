"""SQM module."""

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
from opensqm.rdkit_utils import crop_and_cap_protein, set_residue_info

project_dir = Path("data/inputs/PL-REX/")

target_name = "003-CK2"

target_dir = project_dir / target_name

logger.remove()
logger.add(sys.stderr, level="INFO")


def run_sqm(*, protein_file: Path, ligand_file: Path):
    """Run SQM on protein and ligand."""
    print(protein_file, ligand_file)
    ligand = Chem.MolFromMolFile(str(ligand_file), removeHs=False)

    protein = Chem.MolFromPDBFile(
        str(protein_file), removeHs=False, proximityBonding=True, sanitize=False
    )

    unipka = UnipKa()
    ligand_no_h = Chem.RemoveHs(ligand)
    dist_df = unipka.get_distribution(ligand_no_h, pH=7.4)
    G_Hplus = dist_df[dist_df.is_query_mol].relative_ph_adjusted_free_energy.iloc[0]
    G_Hplus = float(G_Hplus)

    ligand = set_residue_info(ligand)
    ligand = fix_nitro_groups(ligand)

    Chem.rdmolops.Kekulize(protein)
    Chem.rdmolops.Kekulize(ligand)

    print(protein.GetNumAtoms())

    Chem.MolToPDBFile(protein, "/tmp/prot.pdb")

    protein = crop_and_cap_protein(protein=protein, ligand=ligand, distance_to_ligand=12)
    # capped_protein = Chem.AddHs(protein, addCoords=True, onlyOnAtoms=cap_ids)

    Chem.MolToPDBFile(protein, "/tmp/pocket.pdb")

    print(protein.GetNumAtoms())

    complex = Complex(protein=protein, ligand=ligand)

    # AllChem.MMFFOptimizeMolecule(mmff_mol)
    # print(AllChem.GetBestRMS(mmff_mol, mopac_mol))

    # complex.ligand, complex.protein = optimise_complex(
    #     ligand=complex.ligand,
    #     protein=complex.protein,
    #     mode="ligand",
    #     gnorm=5,
    #     num_epochs=10,
    #     use_rapid=True,
    # )
    # complex.ligand, complex.protein = optimise_complex(
    #     ligand=complex.ligand,
    #     protein=complex.protein,
    #     mode="ligand",
    #     gnorm=1,
    #     num_epochs=10,
    #     use_rapid=False,
    # )

    # ligand_charge = Chem.GetFormalCharge(ligand)
    # ligand_free = run_opt_from_rdmol(ligand, mopac_keywords=["PM6-D3H4X", "EPS=78.5"], charge=ligand_charge)

    # E_ligand_bound = run_singlepoint_from_rdmol(ligand, use_mozyme=True, solvent="cosmo2", charge=ligand_charge)
    # E_ligand_free = run_singlepoint_from_rdmol(ligand_free, use_mozyme=True, solvent="cosmo2", charge=ligand_charge)
    # dE_ligand = E_ligand_bound - E_ligand_free

    scores = run_interaction_energy(ligand=complex.ligand, protein=complex.protein)

    scores["G_Hplus"] = G_Hplus
    # scores["dE_ligand"] = dE_ligand

    scores["score"] = scores["dE_int"] + G_Hplus

    print(scores)

    return scores


if __name__ == "__main__":
    scores = []
    df = pd.read_csv(target_dir / "protein.csv")

    for i, row in tqdm(enumerate(df.itertuples()), total=len(df)):
        _scores = run_sqm(protein_file=row.protein, ligand_file=row.ligand)  # type: ignore
        scores.append(_scores)
        scores_df = pd.DataFrame(scores)

        if len(scores_df) == 1:
            continue

        # import pdb; pdb.set_trace()
        corr = scipy.stats.spearmanr(scores_df["score"], df.head(i + 1)["pX"])
        print(corr)

    scores_df.to_csv(target_dir / "scores.csv", index=False)
