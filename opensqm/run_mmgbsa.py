"""Run Molecular Dynamics simulation."""

import itertools
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger
from openmm import unit
from openmm.app import Modeller
from openmm.app.pdbfile import PDBFile
from pydantic import BaseModel
from rdkit import Chem, RDLogger

from tqdm import tqdm

from opensqm.md.mmgbsa import MMGBSASettings, get_interaction_energy
from opensqm.md.prepare import prepare_complex
from opensqm.md.representative import get_representative_frame
from opensqm.md.vanilla import ProductionSettings, production
from opensqm.rdkit_utils import set_coordinates
from opensqm.torsion_scanner import autodetect_flip_dihedrals
from opensqm.md.equilibrate import EquilibrationSettings, equilibrate
from pydantic import ConfigDict
RDLogger.DisableLog('rdApp.warning')



class MMGBSASettings(BaseModel):
    """Settings for MMGBSA interaction energy calculation."""

    model_config = ConfigDict(frozen=True)
    n_closest_waters: int = 5
    n_replicas: int = 1
    ligand_resname: str = "LIG" 
    equilibration_config: EquilibrationSettings = EquilibrationSettings()
    production_config: ProductionSettings = ProductionSettings()  
    overwrite: bool = False
    use_terminal_ring_mc: bool = True



def run_mmgbsa(
    protein: str,
    ligand: str,
    output: str,
    config: MMGBSASettings = MMGBSASettings(),

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
    protein_pdb = PDBFile(protein)

    if config.use_terminal_ring_mc:
        bonds = autodetect_flip_dihedrals(ligand_rdmol)
        print(bonds)
    else:
        bonds = []

    protein_modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    topology, positions, forcefield = prepare_complex(
        ligand_rdmol, padding=1.2, protein_modeller=protein_modeller
    )
    terminal_dihedrals = list(bonds)
    PDBFile.writeFile(topology, positions, open(topology_path, "w"), keepIds=True)
    logger.info("Equilibrating complex")

    if equil_pdb_path.exists() and not config.overwrite:
        logger.info(f"Equilibrated complex already exists")
    else:
        logger.info(f"Equilibrating complex")
        topology, positions = equilibrate(topology, positions, forcefield, config=config.equilibration_config)
        PDBFile.writeFile(topology, positions, open(equil_pdb_path, "w"), keepIds=True)

    all_trajs: list[str] = []
    all_energies: list[list[float] | np.ndarray] = []
    all_tops: list[str] = []
    for replica_i in tqdm(range(config.n_replicas)):
        replica_dir = output_path / f"replica_{replica_i}"
        replica_dir.mkdir(exist_ok=True, parents=True)

        trajectory_path = replica_dir / "com.dcd"
        close_trajectory_path = replica_dir / "com.close.dcd"
        close_top_path = replica_dir / "com.close.prmtop"
        energies_path = replica_dir / "energies.csv"

        
        if trajectory_path.exists() and not config.overwrite:
            logger.info(f"Production simulation for replica {replica_i} already exists")
        else:
            logger.info(f"Running production simulation for replica {replica_i}")
           
            production(
                topology,
                positions,
                forcefield,
                trajectory_path,
                config=config.production_config,
                terminal_dihedrals=terminal_dihedrals,
            )
            

        print(equil_pdb_path, trajectory_path, close_top_path, close_trajectory_path)

        energies, rmsd, close_top_path, close_traj_path = get_interaction_energy(
            pdb_path=str(equil_pdb_path),
            traj_path=str(trajectory_path),
            close_traj_path=str(close_trajectory_path),
            close_top_path=str(close_top_path),
            ligand_path=ligand,
            n_closest_waters=config.n_closest_waters,
            ligand_resname=config.ligand_resname,
        )

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

    prot_modeller = Modeller(positions=modeller.positions, topology=modeller.topology)
    prot_modeller.delete([a for a in prot_modeller.topology.atoms() if a.residue.name == "LIG"])

    lig_modeller = Modeller(positions=modeller.positions, topology=modeller.topology)
    lig_modeller.delete([a for a in lig_modeller.topology.atoms() if a.residue.name != "LIG"])

    ligand_coords = np.array(lig_modeller.positions.value_in_unit(unit.angstrom))

    ligand_rdmol = set_coordinates(ligand_rdmol, coords=ligand_coords)

    PDBFile.writeFile(
        prot_modeller.topology, prot_modeller.positions, open(representative_prot_path, "w"), keepIds=True
    )

    Chem.MolToMolFile(ligand_rdmol, str(representative_lig_path))

    scores = pd.Series(scores)
    scores.to_csv(score_path, header=False)

    logger.info(f"Saved scores to {score_path}")
    logger.info(f"Saved representative protein to {representative_prot_path}")
    logger.info(f"Saved representative ligand to {representative_lig_path}")


    return representative_prot_path, representative_lig_path, scores


@click.command()
@click.option(
    "--protein",
    required=True,
)
@click.option(
    "--ligand",
    required=True,
)
@click.option(
    "--output",
    required=True,
)
@click.option("--n_replicas", default=1)
@click.option(
    "--run-time",
    default=0.1,
)
@click.option("--n_closest_waters", default=5)
@click.option("--overwrite", is_flag=True)
def main(
    protein: str,
    ligand: str,
    output: str,
    n_replicas: int,
    run_time: float,
    n_closest_waters,
    overwrite: bool,
):
    """Run MD and MMGBSA for a single protein ligand complex."""
    print("Running MD and MMGBSA for a single protein ligand complex.")
    print("Protein: ", protein)
    print("Ligand: ", ligand)
    print("Output: ", output)
    print("N_replicas: ", n_replicas)
    print("Run-time: ", run_time)
    print("N_closest_waters: ", n_closest_waters)
    print("Overwrite: ", overwrite)
    scores = run_mmgbsa(
        protein,
        ligand,
        output,
        n_replicas,
        run_time,
        overwrite,
        mmgbsa_config=MMGBSASettings(n_closest_waters=n_closest_waters),
    )
    print(scores)


if __name__ == "__main__":
    main()
