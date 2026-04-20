"""Run Molecular Dynamics simulation."""

from opensqm.md.mmgbsa import get_interaction_energy

equil_pdb_path = "data/outputs/PL-REX/003-CK2/1F0Q/equil.pdb"
trajectory_path = "data/outputs/PL-REX/003-CK2/1F0Q/replica_0/com.dcd"
close_trajectory_path = "data/outputs/PL-REX/003-CK2/1F0Q/replica_0/com.close.dcd"
close_top_path = "data/outputs/PL-REX/003-CK2/1F0Q/replica_0/com.close.prmtop"
ligand = "data/inputs/PL-REX/003-CK2/1F0Q.sdf"
n_closest_waters = 5

energies, rmsd, close_top_path, close_traj_path = get_interaction_energy(
    pdb_path=str(equil_pdb_path),
    traj_path=str(trajectory_path),
    close_traj_path=str(close_trajectory_path),
    close_top_path=str(close_top_path),
    ligand_path=ligand,
    n_closest_waters=n_closest_waters,
)
