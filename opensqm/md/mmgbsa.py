"""Module for calculating MMGBSA interaction energies."""

from typing import Any

import mdtraj as md  # type: ignore
import numpy as np
import pytraj  # type: ignore
from openff.toolkit.topology import Molecule  # type: ignore
from openff.toolkit.utils.toolkits import AmberToolsToolkitWrapper  # type: ignore
from openmm import Context, LangevinMiddleIntegrator, unit  # type: ignore
from openmm.app import CutoffNonPeriodic, ForceField, HBonds, Modeller, PDBFile  # type: ignore
from openmmforcefields.generators import SMIRNOFFTemplateGenerator  # type: ignore
from pymbar import timeseries
from rdkit import Chem
from tqdm import tqdm


# --- Create Systems for complex, protein, ligand ---
def make_system(topology: Any, forcefield: ForceField) -> Any:
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
        # implicitSolvent=OBC2,
        # soluteDielectric=1.0,
        # solventDielectric=80.0
    )
    return system


def get_interaction_energy(
    pdb_path: str,
    ligand_path: str,
    traj_path: str,
    close_traj_path: str,
    close_top_path: str,
    n_closest_waters: int = 5,
    ligand_resname: str = "LIG",
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

    print(complex.topology.getPeriodicBoxVectors())

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
    ligand_rdmol = Chem.AddHs(Chem.MolFromMolFile(ligand_path, removeHs=False), addCoords=True)

    # Image in mdtraj
    ref = md.load(str(pdb_path))
    traj = md.load(str(traj_path), top=str(pdb_path))
    protein_idxs = ref.topology.select("protein and name CA")
    traj = traj.image_molecules()
    traj = traj.superpose(traj[0], atom_indices=protein_idxs, ref_atom_indices=protein_idxs)
    traj.save(str(traj_path))

    # Get closest in pytraj
    traj = pytraj.load(str(traj_path), top=str(pdb_path))
    traj = pytraj.center(traj)  # centers protein in box
    traj = traj.strip(":NA,CL")  # remove ions
    traj = pytraj.align(traj, ref=0, mask="@CA")  # optional alignment
    traj_closest = pytraj.closest(
        traj,
        mask=f":{ligand_resname}",
        n_solvents=n_closest_waters,
        dtype="trajectory",
        top=traj.topology,
    )

    rmsd = pytraj.rmsd(traj, mask=f":{ligand_resname}", ref=0)

    traj_closest.top.save(str(close_top_path))
    pytraj.write_traj(str(close_traj_path), traj_closest, overwrite=True)

    lig_mask = traj_closest.top.select(f":{ligand_resname}")
    prot_mask = traj_closest.top.select(f"!:{ligand_resname}")

    offmol = Molecule.from_rdkit(ligand_rdmol, allow_undefined_stereo=False)
    offmol.assign_partial_charges(
        "am1bcc", toolkit_registry=AmberToolsToolkitWrapper(), use_conformers=offmol.conformers
    )

    smirnoff = SMIRNOFFTemplateGenerator(
        forcefield="openff-2.2.0.offxml", molecules=offmol, cache="smirnoff.json"
    )
    files = [
        "amber/ff14SB.xml",
        "amber/phosaa10.xml",
        "amber/tip3p_standard.xml",
        "implicit/obc1.xml",
    ]

    forcefield_complex = ForceField(*files)
    forcefield_complex.registerTemplateGenerator(smirnoff.generator)

    forcefield_protein = ForceField(*files)

    forcefield_ligand = ForceField("implicit/obc1.xml")
    forcefield_ligand.registerTemplateGenerator(smirnoff.generator)

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
    for frame in tqdm(traj_closest):
        pos = frame.xyz * unit.angstrom

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
    t0, g, Neff_max = timeseries.detect_equilibration(energy)

    print("Neff_max", Neff_max)
    energy_eq = energy[t0:]
    traj_closest_eq = traj_closest[t0:]
    idx = timeseries.subsample_correlated_data(energy_eq, g=g)
    energy = energy_eq[idx]
    traj_closest = traj_closest_eq[idx]
    pytraj.write_traj(str(close_traj_path), traj_closest, overwrite=True)

    return energies, rmsd, close_top_path, close_traj_path


if __name__ == "__main__":
    energies = get_interaction_energy(
        pdb_path="data/trajectories/Glue/CDC34-UBB/UM0131538/replica_0/com.pdb",
        traj_path="data/trajectories/Glue/CDC34-UBB/UM0131538/replica_0/com.nc",
        ligand_path="data/trajectories/Glue/CDC34-UBB/UM0131538/replica_0/lig.sdf",
        close_traj_path="data/trajectories/Glue/CDC34-UBB/UM0131538/replica_0/close_traj.nc",
        close_top_path="data/trajectories/Glue/CDC34-UBB/UM0131538/replica_0/close_top.prmtop",
    )

    print(energies)
