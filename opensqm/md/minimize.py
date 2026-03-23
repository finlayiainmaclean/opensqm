"""Module for running energy minimization on OpenMM Modeller objects."""

from openmm import LangevinIntegrator, app, unit  # type: ignore


def minimize(
    modeller: app.Modeller,
    ff_files: tuple[str, ...] = ("amber14-all.xml", "implicit/gbn2.xml", "amber14/tip3pfb.xml"),
) -> app.Modeller:
    """
    Minimize the energy of the given modeller object.

    Parameters
    ----------
    modeller : app.Modeller
        The Modeller object to minimize.
    ff_files : tuple, optional
        Force field files to use.

    Returns
    -------
    app.Modeller
        The minimized Modeller.
    """
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

    modeller = app.Modeller(
        modeller.topology, sim.context.getState(getPositions=True).getPositions()
    )
    return modeller
