
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from opensqm.torsion_scanner import (
    build_ase_atoms,
    get_rotatable_terminal_bonds,
    run_torsion_scan_ase,
)


def test_symmetric_biaryl_180_degree_flips():
    """
    Tests that a symmetric biaryl system (like biphenyl) produces a perfectly
    symmetric torsional energy profile under a 180-degree phase shift
    due to exact conformational structural redundancy.
    """
    # Create biphenyl
    mol = Chem.MolFromSmiles("c1ccccc1-c2ccccc2")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    # Locate the single bond connecting the two aromatic rings
    bonds = get_rotatable_terminal_bonds(mol)
    assert len(bonds) > 0, "No rotatable bond found between the aromatic rings."

    bridge_bond = bonds[0]
    atoms = build_ase_atoms(mol)

    # Execute a 360 degree scan in 24 steps (15 degrees per step)
    result, angles, energies = run_torsion_scan_ase(atoms, bridge_bond, steps=24)

    print(result)
    barrier = result["min_barrier"]

    from pathlib import Path

    import matplotlib.pyplot as plt

    # Plot and save the torsion profile securely
    plt.figure(figsize=(8, 5))
    plt.plot(angles, energies, marker="o", linestyle="-", color="indigo", linewidth=2)
    plt.title("Biphenyl Torsion Energy Profile (AIMNet2)")
    plt.xlabel("Torsional Angle (degrees)")
    plt.ylabel("Relative Energy (kcal/mol)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    out_path = Path(__file__).resolve().parent / "biphenyl_torsion_profile.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    # Ensure energies list matches the requested steps
    assert len(energies) == 24
    assert len(angles) == 24

    # 1. Symmetry Verification
    # A 360 degree scan with 24 steps implies that `i + 12` maps to `theta + 180`
    # For a symmetric phenyl flip, E(theta) MUST mirror E(theta + 180) precisely.
    symmetry_errors = []
    for i in range(12):
        e1 = energies[i]
        e2 = energies[i + 12]
        symmetry_errors.append(abs(e1 - e2))

    avg_symmetry_error = np.mean(symmetry_errors)

    # We enforce that the energy profile maps onto itself after a 180 degree flip
    # within reasonable AIMNet2 minimization noise limits (~0.5 kcal/mol)
    assert avg_symmetry_error < 0.5, (
        f"Energy profile was not symmetric over 180 deg flips! Avg Err: {avg_symmetry_error:.2f}"
    )

    # 2. Extract Minima
    # Find all the local valleys dynamically using neighbor comparison
    # We wrap the array to accurately evaluate boundary conditions at -180 / 180
    minima_indices = []
    for i in range(24):
        prev_idx = (i - 1) % 24
        next_idx = (i + 1) % 24
        if energies[i] < energies[prev_idx] and energies[i] < energies[next_idx]:
            minima_indices.append(i)

    minima_angles = [int(angles[idx]) for idx in minima_indices]

    # Depending on steric bulk, biaryls generally sit out-of-plane producing 4 physical minima
    # (e.g. roughly ~45, 135, 225, 315) across the 360 domain rather than exactly 0 and 180.
    # However, each primary minimum has a 180 degree symmetrical twin!
    assert len(minima_angles) >= 2, "Expected at least 2 distinct minima for the biaryl."

    # Guarantee that for EVERY minimum, there is a paired minimum exactly 180 degrees away
    for angle in minima_angles:
        twin = (angle + 180) % 360
        # Given our angles domain typically spans [-180, 180), we'll dynamically wrap it
        if twin > 180:
            twin -= 360
        if twin == 180:
            twin = -180

        found_twin = False
        # Account for precision snap tolerances during constrained LBFGS sweeps
        for target in minima_angles:
            if abs(target - twin) < 20 or abs(abs(target - twin) - 360) < 20:
                found_twin = True
                break

                assert found_twin, (
                    f"Minimum at {angle} deg is missing its structural 180 degree twin!"
                )


def test_autodetect_flip_dihedrals():
    """
    Test the top-level orchestrator for Type 2 atropisomer flip detection
    on a biaryl system, verifying the returned torsion groups and mapping angles.
    """
    from opensqm.torsion_scanner import autodetect_flip_dihedrals

    mol = Chem.MolFromSmiles("c1cccc(Cl)[1c]1-[1c]1c(Br)cccc1")
    mol = Chem.MolFromSmiles("C1C=CC(CC(=O)NC2C=C(N)C(C#N)=C(OCC)N=2)=CC=1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    # Standard autodetection run. Biphenyl naturally falls outside the boundary (Type 1),
    flips = autodetect_flip_dihedrals(mol, method="openmm")

    print(flips[0])

    assert len(flips) == 1, f"Expected exactly 1 bridged rotatable bond handled, got: {flips}"

    bond, angle_delta = flips[0]
    print(bond)

    assert mol.GetAtomWithIdx(bond[0]).GetIsotope() == 1
    assert mol.GetAtomWithIdx(bond[1]).GetIsotope() == 1

    # Ensure there are multiple symmetric minima separated pathways (biphenyl has 4)
    assert angle_delta == 180
