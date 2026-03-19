"""SQM module."""

import sys
from pathlib import Path

import pandas as pd
import ray
import scipy
import xxhash
from loguru import logger
from pydantic import BaseModel, ConfigDict
from rdkit import Chem
from tqdm.auto import tqdm
from unipka import UnipKa

from opensqm._types import Complex
from opensqm.mopac import (
    fix_nitro_groups,
    optimise_complex,
    run_interaction_energy,
    run_opt_from_rdmol,
    run_singlepoint_from_rdmol,
)
from opensqm.rdkit_utils import crop_and_cap_protein, set_residue_info

dataset = "PL-REX"
target_name = "003-CK2"
input_dir = Path(f"data/inputs/{dataset}")
input_target_dir = input_dir / target_name
output_dir = Path(f"data/outputs/{dataset}")
output_target_dir = output_dir / target_name

output_target_dir.mkdir(exist_ok=True, parents=True)

logger.remove()
logger.add(sys.stderr, level="INFO")


class SqmInput(BaseModel):
    """Paths to protein PDB and ligand MOL for one SQM run."""

    model_config = ConfigDict(frozen=True)
    complex_id: str
    protein_file: Path
    ligand_file: Path
    optimise: bool = True

    def __hash__(self) -> int:
        """xxHash-64 of complex id and both paths (POSIX), in order."""
        h = xxhash.xxh64()
        h.update(self.complex_id.encode("utf-8"))
        h.update(b"\0")
        for path in (self.protein_file, self.ligand_file):
            h.update(path.expanduser().as_posix().encode("utf-8"))
            h.update(b"\0")
        return h.intdigest()


class SqmOutput(BaseModel):
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
def run_sqm_wrapper(inp: SqmInput) -> SqmOutput:
    """Run SQM on a protein and ligand with caching."""
    output_json = output_target_dir / f"{hash(inp)}.json"

    if output_json.exists():
        print("Output already exists")
        return SqmOutput.model_validate_json(output_json.read_text())
    else:
        print("Running SQM")
        output = run_sqm(inp)
        output_json.write_text(output.model_dump_json())
        return output


def run_sqm(inp: SqmInput) -> SqmOutput:
    """Run SQM on protein and ligand."""
    print(inp.protein_file, inp.ligand_file)
    ligand = Chem.MolFromMolFile(str(inp.ligand_file), removeHs=False)

    protein = Chem.MolFromPDBFile(
        str(inp.protein_file), removeHs=False, proximityBonding=True, sanitize=False
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

    protein = crop_and_cap_protein(protein=protein, ligand=ligand, distance_to_ligand=10)
    # capped_protein = Chem.AddHs(protein, addCoords=True, onlyOnAtoms=cap_ids)

    Chem.MolToPDBFile(protein, "/tmp/pocket.pdb")

    print(protein.GetNumAtoms())

    complex = Complex(protein=protein, ligand=ligand)

    # AllChem.MMFFOptimizeMolecule(mmff_mol)
    # print(AllChem.GetBestRMS(mmff_mol, mopac_mol))

    complex.ligand, complex.protein = optimise_complex(
        ligand=complex.ligand,
        protein=complex.protein,
        mode="ligand",
        gnorm=20,
        num_epochs=5,  # ~4 minutes an epoch
        use_rapid=True,
    )
    complex.ligand, complex.protein = optimise_complex(
        ligand=complex.ligand,
        protein=complex.protein,
        mode="ligand",
        gnorm=10,
        num_epochs=3,  # # ~5 minutes an epoch
        use_rapid=False,
    )

    ligand_charge = Chem.GetFormalCharge(ligand)
    E_ligand_bound = run_singlepoint_from_rdmol(
        complex.ligand, use_mozyme=True, solvent="cosmo2", charge=ligand_charge
    )

    ligand_free = run_opt_from_rdmol(
        complex.ligand, mopac_keywords=["PM6-D3H4X", "EPS=78.5"], charge=ligand_charge
    )

    E_ligand_free = run_singlepoint_from_rdmol(
        ligand_free, use_mozyme=True, solvent="cosmo2", charge=ligand_charge
    )

    dE_ligand_strain = E_ligand_bound - E_ligand_free

    scores = run_interaction_energy(ligand=complex.ligand, protein=complex.protein)
    score = scores["dE_int"] + G_Hplus + dE_ligand_strain

    print(
        f"dE_int: {scores['dE_int']}, G_Hplus: {G_Hplus}, "
        f"dE_ligand_strain: {dE_ligand_strain}, score: {score}"
    )

    return SqmOutput(
        dE_int=scores["dE_int"],
        E_complex=scores["E_complex"],
        E_protein=scores["E_protein"],
        E_ligand=scores["E_ligand"],
        G_Hplus=G_Hplus,
        dE_ligand_strain=dE_ligand_strain,
        score=score,
    )


if __name__ == "__main__":
    ray.init(num_cpus=5)

    df = pd.read_csv(input_target_dir / "protein.csv")
    df = df.dropna(subset=["protein"])
    futures = [
        run_sqm_wrapper.remote(
            SqmInput(
                complex_id=row.id,
                protein_file=Path(row.protein),
                ligand_file=Path(row.ligand),
                optimise=True,
            )
        )
        for row in tqdm(df.itertuples(), total=len(df), desc="Submitting")
    ]
    results = ray.get(futures)

    scores_df = pd.DataFrame([r.model_dump() for r in results])
    if len(scores_df) > 1:
        corr = scipy.stats.spearmanr(scores_df["score"], df["pX"])
        print(corr)
    scores_df.to_csv(output_target_dir / "scores.csv", index=False)
