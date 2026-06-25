"""Module for calculating MMGBSA interaction energies."""

import copy
from typing import Any

import mdtraj as md
import numpy as np
from openff.toolkit.topology import Molecule  # type: ignore
from openmm import Context, LangevinMiddleIntegrator, unit
from openmm.app import (
    CutoffNonPeriodic,
    ForceField,
    HBonds,
    Modeller,
    PDBFile,
    Topology,
)
from pymbar import timeseries
from rdkit import Chem
from scipy.spatial.distance import cdist
from tqdm import tqdm

from opensqm.md.prepare import get_ligand_forcefield


def _openmm_atom_lookup_key(atom: Any) -> tuple[Any, ...]:
    rid = atom.residue.id
    try:
        rid_i = int(rid)
    except (TypeError, ValueError):
        rid_i = rid
    return (str(atom.residue.chain.id).strip(), rid_i, atom.residue.name.strip(), atom.name.strip())


def _mdtraj_atom_lookup_key(atom: Any) -> tuple[Any, ...]:
    ch = atom.residue.chain
    cid = getattr(ch, "chain_id", None)
    if cid is None:
        cid = str(ch.index)
    return (
        str(cid).strip(),
        int(atom.residue.resSeq),
        atom.residue.name.strip(),
        atom.name.strip(),
    )


# --- Create Systems for complex, protein, ligand ---
def make_system(topology: Topology, forcefield: ForceField) -> Any:
    """
    Create an OpenMM System from the given topology and forcefield.

    Parameters
    ----------
    topology : openmm.app.Topology
        The topology.
    forcefield : openmm.app.ForceField
        The forcefield.

    Returns
    -------
    openmm.System
        The created system.
    """
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=CutoffNonPeriodic,
        constraints=HBonds,
    )

    return system


def calculate_average_waters_per_residue(
    traj: md.Trajectory, cutoff: float = 0.4
) -> dict[int, float]:
    """
    Calculate the average number of water molecules within a cutoff distance
    of each protein residue over a trajectory.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory, which must contain both protein and water.
    cutoff : float, optional
        The distance cutoff in nanometers (default 0.4 nm = 4 Angstroms).

    Returns
    -------
    dict[int, float]
        Dictionary mapping the mdtraj residue index to the average number of
        water molecules within the cutoff across all frames.
    """
    protein_residues = [r for r in traj.topology.residues if r.is_protein]

    # Get oxygen atoms for waters
    water_atoms = []
    for r in traj.topology.residues:
        if r.name in ("HOH", "WAT", "TIP3", "TIP4", "TIP5"):
            for a in r.atoms:
                if a.element.symbol == "O":
                    water_atoms.append(a.index)
                    break
    water_atoms = np.array(water_atoms)

    avg_waters = {}
    for res in protein_residues:
        # Use heavy atoms of the residue for distance calculation
        res_atoms = np.array([a.index for a in res.atoms if a.element.symbol != "H"])
        if len(res_atoms) == 0:
            continue

        # Find waters within cutoff of any heavy atom in this residue
        neighbors = md.compute_neighbors(traj, cutoff, res_atoms, haystack_indices=water_atoms)
        counts = [len(frame_neighbors) for frame_neighbors in neighbors]
        avg_waters[res.index] = float(np.mean(counts))

    return avg_waters


def get_interaction_energy(
    pdb_path: str,
    ligand_path: str,
    traj_path: str,
    close_traj_path: str,
    close_top_path: str,
    n_closest_waters: int = 5,
    ligand_resname: str = "LIG",
    frame_indices: np.ndarray | None = None,
    offmol: "Molecule | None" = None,
) -> tuple[list[float], np.ndarray, str, str]:
    """
    Calculate the MMGBSA interaction energy.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.
    ligand_path : str
        Path to the ligand SDF file.
    traj_path : str
        Path to the trajectory file.
    close_traj_path : str
        Path where the closest trajectory will be saved.
    close_top_path : str
        Path where the closest topology will be saved.
    n_closest_waters : int, optional
        Number of closest waters to keep, by default 5.
    ligand_resname : str, optional
        Residue name of the ligand, by default "LIG".

    Returns
    -------
    tuple
        Energies, RMSD, topology path, trajectory path.
    """
    complex = PDBFile(str(pdb_path))

    # Find all water residues
    modeller = Modeller(positions=complex.positions, topology=complex.topology)
    water_residues = []
    ion_residues = []
    for residue in modeller.topology.residues():
        if residue.name in ("HOH", "WAT", "TIP3", "TIP4", "TIP5"):
            water_residues.append(residue)
        elif residue.name in ("NA", "CL"):
            ion_residues.append(residue)

    waters_to_delete = water_residues[n_closest_waters:]
    to_delete = waters_to_delete + ion_residues

    modeller.delete(to_delete)
    if offmol is None:
        ligand_rdmol = Chem.AddHs(Chem.MolFromMolFile(ligand_path, removeHs=False), addCoords=True)
        offmol = Molecule.from_rdkit(ligand_rdmol, allow_undefined_stereo=True)

    # Image in mdtraj
    imaged_traj_path = str(traj_path).replace(".dcd", "_imaged.dcd")
    ref = md.load(str(pdb_path))
    traj_md = md.load(str(traj_path), top=str(pdb_path))
    if frame_indices is not None:
        traj_md = traj_md[frame_indices]
    protein_idxs = ref.topology.select("protein and name CA")
    traj_md = traj_md.image_molecules()
    traj_md = traj_md.superpose(
        traj_md[0], atom_indices=protein_idxs, ref_atom_indices=protein_idxs
    )
    traj_md.save(imaged_traj_path)

    # Clean up protein/ligand
    # We want: protein, the ligand, and all waters.
    water_res = [
        r for r in ref.topology.residues if r.name in ("HOH", "WAT", "TIP3", "TIP4", "TIP5")
    ]
    water_atom_idxs = np.array([[a.index for a in r.atoms] for r in water_res])

    ligand_idxs = ref.topology.select(f"resname {ligand_resname}")
    rmsd = md.rmsd(traj_md, traj_md[0], atom_indices=ligand_idxs)

    # Map each OpenMM modeller atom (protein → retained waters → ligand) to a static
    # mdtraj index in the full trajectory. Waters use -1 and are filled per-frame from
    # the closest n water molecules so coordinates match modeller.topology order.
    ref_key_to_idx = {_mdtraj_atom_lookup_key(a): a.index for a in ref.topology.atoms}
    n_omm = modeller.topology.getNumAtoms()
    ref_by_omm = np.full(n_omm, -1, dtype=np.int64)
    for omm_atom in modeller.topology.atoms():
        if omm_atom.residue.name in ("HOH", "WAT", "TIP3", "TIP4", "TIP5"):
            continue
        ref_by_omm[omm_atom.index] = ref_key_to_idx[_openmm_atom_lookup_key(omm_atom)]
    water_omm_indices = np.flatnonzero(ref_by_omm < 0)
    if (
        len(water_atom_idxs) > 0
        and len(water_omm_indices) != n_closest_waters * water_atom_idxs.shape[1]
    ):
        raise ValueError(
            "Water atom count in modeller does not match "
            "n_closest_waters * atoms per water residue."
        )

    closest_xyz = []

    if len(water_atom_idxs) > 0:
        water_O_idxs = water_atom_idxs[:, 0]
        static_mask = ref_by_omm >= 0
        for i in range(traj_md.n_frames):
            full = traj_md.xyz[i]
            frame_xyz = np.empty((n_omm, 3), dtype=np.float64)
            frame_xyz[static_mask] = full[ref_by_omm[static_mask]]

            lig_coords = full[ligand_idxs, :]
            wat_coords = full[water_O_idxs, :]
            dists = cdist(wat_coords, lig_coords)
            min_dists = dists.min(axis=1)
            closest_wat = np.argsort(min_dists)[:n_closest_waters]
            closest_flat = water_atom_idxs[closest_wat].flatten()
            frame_xyz[water_omm_indices] = full[closest_flat, :]
            closest_xyz.append(frame_xyz)
    else:
        for i in range(traj_md.n_frames):
            closest_xyz.append(traj_md.xyz[i][ref_by_omm, :])

    closest_xyz = np.array(closest_xyz)

    # Save the processed topology and DCD using MDTraj directly
    openmm_top_mdtraj = md.Topology.from_openmm(modeller.topology)
    traj_closest = md.Trajectory(xyz=closest_xyz, topology=openmm_top_mdtraj)

    # Save a PDB as the topology and DCD for the trajectory
    traj_closest[0].save_pdb(str(close_top_path).replace(".prmtop", ".pdb"))
    traj_closest.save_dcd(str(close_traj_path))

    lig_mask = traj_closest.topology.select(f"resname {ligand_resname}")
    prot_mask = traj_closest.topology.select(f"not resname {ligand_resname}")

    forcefield_complex = get_ligand_forcefield(offmol, True)
    forcefield_ligand = copy.deepcopy(forcefield_complex)
    forcefield_ligand.loadFile("implicit/obc1.xml")

    files = (
        "amber/ff14SB.xml",
        "amber/phosaa10.xml",
        "amber/tip3p_HFE_multivalent.xml",
        "amber/tip3p_standard.xml",
        "implicit/obc1.xml",
    )
    forcefield_complex.loadFile(files)

    forcefield_protein = ForceField(*files)

    complex = Modeller(positions=modeller.positions, topology=modeller.topology)
    protein = Modeller(positions=modeller.positions, topology=modeller.topology)
    ligand = Modeller(positions=modeller.positions, topology=modeller.topology)

    protein.delete([a for a in complex.topology.atoms() if a.residue.name == ligand_resname])
    ligand.delete([a for a in complex.topology.atoms() if a.residue.name != ligand_resname])

    system_ligand = make_system(ligand.topology, forcefield_ligand)
    system_protein = make_system(protein.topology, forcefield_protein)
    system_complex = make_system(complex.topology, forcefield_complex)


    integrator_complex = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )
    integrator_protein = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )
    integrator_ligand = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )

    context_complex = Context(system_complex, integrator_complex)
    context_protein = Context(system_protein, integrator_protein)
    context_ligand = Context(system_ligand, integrator_ligand)

    energies = []
    for frame_xyz in tqdm(closest_xyz):
        pos = frame_xyz * unit.nanometers

        # Set positions for each system
        context_complex.setPositions(pos)
        E_complex = (
            context_complex.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilocalories_per_mole)
        )

        context_protein.setPositions(pos[prot_mask])
        E_protein = (
            context_protein.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilocalories_per_mole)
        )

        context_ligand.setPositions(pos[lig_mask])
        E_ligand = (
            context_ligand.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilocalories_per_mole)
        )

        dG = E_complex - (E_protein + E_ligand)
        energies.append(dG)

    energy = np.array(energies)

    # Find equilibrated and decorrelated frames/energies
    t0, g, _Neff_max = timeseries.detectEquilibration(energy)

    energy_eq = energy[t0:]
    closest_xyz_eq = closest_xyz[t0:]
    idx = timeseries.subsampleCorrelatedData(energy_eq, g=g)
    energy = energy_eq[idx]

    # Update the final exported closest trajectory with decorrelated frames
    final_eq_xyz = closest_xyz_eq[idx]
    traj_final = md.Trajectory(xyz=final_eq_xyz, topology=openmm_top_mdtraj)
    traj_final.save_dcd(str(close_traj_path))

    return energies, rmsd, str(close_top_path).replace(".prmtop", ".pdb"), str(close_traj_path)


if __name__ == "__main__":
    from opensqm.md.prepare import prepare_complex

    ligand = "data/inputs/PL-REX/003-CK2/1F0Q.sdf"
    protein = "data/inputs/PL-REX/003-CK2/1F0Q.prot.fixed.pdb"

    ligand_rdmol = Chem.MolFromMolFile(ligand, removeHs=False)
    protein = PDBFile(protein)

    topology, positions, forcefield = prepare_complex(
        ligand_rdmol, protein, bespoke_ligand_forcefield=True
    )

    PDBFile.writeFile(topology, positions, open("/tmp/com.pdb", "w"), keepIds=True)

    energies, rmsd, close_top_path, close_traj_path = get_interaction_energy(
        ligand_path=ligand,
        pdb_path="/tmp/com.pdb",
        traj_path="/tmp/com.pdb",
        close_traj_path="/tmp/com.close.dcd",
        close_top_path="/tmp/com.close.prmtop",
    )
    print(energies)
    print(rmsd)
