"""Run Molecular Dynamics simulation."""

import itertools
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger
from openmm import unit  # type: ignore
from openmm.app import Modeller  # type: ignore
from openmm.app.pdbfile import PDBFile  # type: ignore
from rdkit import Chem
from tqdm import tqdm

from opensqm.md.mmgbsa import get_interaction_energy
from opensqm.md.representative import get_representative_frame
from opensqm.md.vanilla import equilibrate, prepare_complex, production
from opensqm.rdkit_utils import set_coordinates


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
@click.option("--n_replicas", default=2)
@click.option(
    "--run-time",
    default=1.0,
)
@click.option("--n_closest_waters", default=5)
def main(
    protein: str,
    ligand: str,
    output: str,
    n_replicas: int,
    run_time: float,
    n_closest_waters,
):
    """Run entrypoint for Molecular Dynamics simulation."""
    output_path = Path(output)
    output_path.mkdir(exist_ok=True, parents=True)

    topology_path = output_path / "com.pdb"
    equil_pdb_path = output_path / "equil.pdb"
    representative_prot_path = output_path / "prot.pdb"
    representative_lig_path = output_path / "lig.sdf"
    score_path = output_path / "scores.csv"

    ligand_rdmol = Chem.AddHs(Chem.MolFromMolFile(ligand, removeHs=False), addCoords=True)
    protein = PDBFile(protein)

    topology, positions, forcefield = prepare_complex(ligand_rdmol, protein)
    PDBFile.writeFile(topology, positions, open(topology_path, "w"), keepIds=True)
    logger.info("Equilibrating complex")
    topology, positions = equilibrate(topology, positions, forcefield, npt_ps=30, warmup_ps=10)
    PDBFile.writeFile(topology, positions, open(equil_pdb_path, "w"), keepIds=True)

    all_trajs = []
    all_energies = []
    all_tops = []
    for replica_i in tqdm(range(n_replicas)):
        replica_dir = output_path / f"replica_{replica_i}"
        replica_dir.mkdir(exist_ok=True, parents=True)

        trajectory_path = replica_dir / "com.dcd"
        close_trajectory_path = replica_dir / "com.close.dcd"
        close_top_path = replica_dir / "com.close.prmtop"
        energies_path = replica_dir / "energies.csv"

        logger.info(f"Running production simulation for replica {replica_i}")
        production(
            topology,
            positions,
            forcefield,
            trajectory_path,
            run_time_ps=int(run_time * 1000),
            log_ps=1,
        )

        print(equil_pdb_path, trajectory_path, close_top_path, close_trajectory_path)

        energies, rmsd, close_top_path, close_traj_path = get_interaction_energy(
            pdb_path=str(equil_pdb_path),
            traj_path=str(trajectory_path),
            close_traj_path=str(close_trajectory_path),
            close_top_path=str(close_top_path),
            ligand_path=ligand,
            n_closest_waters=n_closest_waters,
        )

        print(energies)

        logger.info(f"Ligand RMSD: {rmsd.mean()}")

        all_energies.append(energies)
        all_trajs.append(close_traj_path)
        all_tops.append(close_top_path)
        energies = pd.DataFrame(energies, columns=["energy"])
        pd.DataFrame(energies).to_csv(energies_path, index=False)

    modeller, _best_energy = get_representative_frame(all_tops, all_trajs, all_energies)

    scores = {}
    _all_energies = list(itertools.chain(*all_energies))
    scores["mm_energy_mean"] = np.mean(_all_energies)
    scores["mm_energy_std"] = np.std(_all_energies)
    scores["mm_energy_min"] = np.min(_all_energies)

    protein = Modeller(positions=modeller.positions, topology=modeller.topology)
    protein.delete([a for a in protein.topology.atoms() if a.residue.name == "LIG"])

    ligand = Modeller(positions=modeller.positions, topology=modeller.topology)
    ligand.delete([a for a in ligand.topology.atoms() if a.residue.name != "LIG"])

    ligand_coords = np.array(ligand.positions.value_in_unit(unit.angstrom))

    ligand_rdmol = set_coordinates(ligand_rdmol, coords=ligand_coords)

    PDBFile.writeFile(
        protein.topology, protein.positions, open(representative_prot_path, "w"), keepIds=True
    )

    Chem.MolToMolFile(ligand_rdmol, str(representative_lig_path))

    scores = pd.Series(scores)
    scores.to_csv(score_path, header=False)

    logger.info(f"Saved scores to {score_path}")
    logger.info(f"Saved representative protein to {representative_prot_path}")
    logger.info(f"Saved representative ligand to {representative_lig_path}")

    print(scores)


if __name__ == "__main__":
    main()
