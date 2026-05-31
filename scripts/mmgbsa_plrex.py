import pandas as pd
from opensqm.md.mmgbsa import MMGBSASettings
from opensqm.run_md import run_mmgbsa
from pathlib import Path
from scipy.stats import spearmanr

TARGET="003-CK2"

df=pd.read_csv(f"data/inputs/PL-REX/{TARGET}/protein.csv", index_col=0).iloc[:11]

print(df.columns)

base_output_dir = Path(f"data/outputs/PL-REX/{TARGET}/")
base_output_dir.mkdir(exist_ok=True, parents=True)

representative_paths = []

scores = []
for row in df.itertuples():

    output_dir = base_output_dir / f"{row.Index}"

    print(row.rscb_protein, row.ligand)


    prot_path, lig_path, score = run_mmgbsa(
        protein=str(row.rscb_protein),
        ligand=str(row.ligand),
        output=str(output_dir),
        n_replicas=1,
        run_time=1,
        mmgbsa_config=MMGBSASettings(n_closest_waters=5),
    )

    representative_paths.append((prot_path, lig_path, str(row.Index), row.pX))

    score['pX'] = row.pX
    scores.append(score)

    scores_df = pd.DataFrame(scores)

    representative_df = pd.DataFrame(representative_paths)
    representative_df.columns = ["protein", "ligand", "id", "pX"]
    representative_df.to_csv(base_output_dir / "representative.csv", index=False)
    print(spearmanr(scores_df.mm_energy_mean, scores_df.pX))


        
