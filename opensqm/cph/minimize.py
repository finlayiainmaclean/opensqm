"""Energy minimisation of protein termini with fixed original heavy atoms."""

from pathlib import Path

from openmm import LangevinIntegrator, app, unit


def minimize(
    modeller: app.Modeller,
    ff_files: tuple = ("amber14-all.xml", "implicit/gbn2.xml", "amber14/tip3pfb.xml"),
) -> app.Modeller:
    """Minimise added hydrogens/caps while holding original residue atoms fixed."""
    forcefield = app.ForceField(*ff_files)

    original_residues = set()
    for residue in modeller.topology.residues():
        if residue.name not in ["ACE", "NME"]:
            original_residues.add(residue.index)

    # Track existing hydrogens on original residues
    original_hydrogens = set()
    for atom in modeller.topology.atoms():
        if atom.element == app.element.hydrogen and atom.residue.index in original_residues:
            original_hydrogens.add((atom.residue.index, atom.name))

    modeller.addHydrogens(forcefield=forcefield)
    # Delete any new hydrogens that appear on original residues
    hydrogens_to_delete = []
    for atom in modeller.topology.atoms():
        if atom.element == app.element.hydrogen and atom.residue.index in original_residues:
            atom_id = (atom.residue.index, atom.name)
            if atom_id not in original_hydrogens:
                hydrogens_to_delete.append(atom)

    modeller.delete(hydrogens_to_delete)

    system = forcefield.createSystem(modeller.topology)

    for atom in modeller.topology.atoms():
        if atom.residue.name not in ["ACE", "NME"]:
            system.setParticleMass(atom.index, 0.0)

    integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond)
    sim = app.Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy()

    return app.Modeller(modeller.topology, sim.context.getState(getPositions=True).getPositions())


if __name__ == "__main__":
    # Check if we're running in test mode
    import argparse

    parser = argparse.ArgumentParser(
        description="Add capping groups ACE and NME to protein termini. "
        "Remove the hydrogens from the input pdb file before using this script"
    )
    parser.add_argument(
        "-i", dest="in_file", type=str, default="protein_noh.pdb", help="Input PDB file"
    )
    parser.add_argument(
        "-o", dest="out_file", type=str, default="protein_noh_cap.pdb", help="Output PDB file"
    )

    args = parser.parse_args()
    in_file = args.in_file
    out_file = args.out_file

    pdb = app.PDBFile(in_file)

    # Create initial modeller
    modeller = app.Modeller(pdb.topology, pdb.positions)

    modeller = minimize(modeller)

    with Path(out_file).open("w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f, True)
