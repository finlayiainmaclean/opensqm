"""Module for obtaining a representative structural frame from MD trajectories."""

import mdtraj as md  # type: ignore
import numpy as np
from openmm.app import Modeller  # type: ignore
from sklearn.cluster import DBSCAN


def calculate_rmsd_matrix(xyz: np.ndarray) -> np.ndarray:
    """
    Compute pairwise RMSD matrix for prealigned coordinates.

    Parameters.
    ----------
    xyz : np.ndarray, shape (N, M, 3)
        Coordinates (assumed pre-aligned)

    Returns
    -------
    rmsd : np.ndarray, shape (N, N)
        Symmetric RMSD matrix in Å
    """
    N, M, _ = xyz.shape
    # Flatten each frame (N, 3M)
    X = xyz.reshape(N, -1)
    # Compute squared distances efficiently
    d2 = (np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)) / M
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2)


def get_representative_frame(
    all_tops: list[str],
    all_trajs: list[str],
    all_energies: list[float] | np.ndarray,
    lig_resname: str = "LIG",
) -> tuple[Modeller, float]:
    """
    Get the most representative MD frame based on RMSD clustering and lowest energy.

    Parameters
    ----------
    all_tops : list
        List of paths to topologies.
    all_trajs : list
        List of paths to trajectories.
    all_energies : list
        List of energies.
    lig_resname : str
        Residue name of the ligand.

    Returns
    -------
    tuple
        (best_modeller, best_energy) where best_modeller is an openmm Modelling object
        for the representative frame and best_energy is its corresponding energy.
    """
    trajs, energies = [], []
    for top, traj_file, energy_val in zip(all_tops, all_trajs, all_energies, strict=False):
        loaded_energy = np.array(energy_val)
        loaded_traj = md.load(traj_file, top=top)
        energies.append(loaded_energy)
        trajs.append(loaded_traj)

    traj = md.join(trajs)
    energy = np.hstack(energies)

    traj.superpose(traj[0], atom_indices=traj.top.select("protein and name CA"))

    lig_atoms = traj.top.select(f"resname {lig_resname} and not element H")

    rmsd_matrix = calculate_rmsd_matrix(traj.xyz[:, lig_atoms, :])

    # 4. Cluster using DBSCAN on precomputed RMSD
    clusterer = DBSCAN(eps=1.0, min_samples=2, metric="precomputed")

    labels = clusterer.fit_predict(rmsd_matrix)

    valid = labels >= 0
    cluster_ids, counts = np.unique(labels[valid], return_counts=True)
    if len(cluster_ids) == 0:
        raise ValueError("No clusters found — adjust eps or min_samples.")

    print(counts)

    largest_cluster = cluster_ids[counts.argmax()]
    in_cluster = labels == largest_cluster

    # Find frames in largest cluster
    cluster_frames = np.where(in_cluster)[0]

    # Get energies for those frames
    cluster_energies = energy[cluster_frames]

    # Find which cluster frame has minimum energy
    min_energy_idx = cluster_energies.argmin()

    # Get the actual trajectory frame index
    best_idx = cluster_frames[min_energy_idx]

    # Now these are correct
    best_positions = traj.openmm_positions(best_idx)
    best_energy = energy[best_idx]

    best_modeller = Modeller(traj.top.to_openmm(), best_positions)
    return best_modeller, best_energy
