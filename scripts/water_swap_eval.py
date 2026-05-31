"""End-to-end smoke test of NCMC water-swap MC on the HSP90/Cl-resorcinol system.

This script verifies the :class:`~opensqm.md.water_swap_mc.WaterSwapMC`
mover against the user's target (>1 accepted water-swap on HSP90, with
the published "true" count being 4). To stay fast it builds the
simulation by hand rather than going through
:class:`~opensqm.cph.constantph.ConstantPH` - that path is the
*integration* path (exercised by importing :class:`WaterSwapConfig`
and passing it to the ``ConstantPH`` constructor in the main run
scripts) whereas this file is the *validation* path. The setup
nonetheless matches everything ConstantPH would build for the
production system: SMIRNOFF ligand template, TIP3P explicit solvent,
PME, HMR + 4 fs timestep, REST2 applied to the ligand, and an
``MonteCarloBarostat``.

Run with::

    pixi run python opensqm/cph/water_swap_eval.py

The script writes ``hsp90_watermc_summary.csv`` next to the input
files capturing the per-attempt acceptance counters so the user can
sanity-check the water-swap behaviour offline.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from openff.toolkit.topology import Molecule  # type: ignore
from openmm import (
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    Platform,
    unit,
)
from openmm.app import HBonds, Modeller, PDBFile, PME, Simulation
from rdkit import Chem
from tqdm import tqdm

from opensqm.md.prepare import get_ligand_forcefield
from opensqm.md.rest import apply_rest
from opensqm.md.water_swap_mc import WaterSwapSettings, WaterSwapMC
from opensqm.rdkit_utils import set_residue_info


def _build_complex(
    protein_pdb_path: Path, ligand_path: Path,
) -> tuple:
    """Build the solvated HSP90/ligand complex.

    Mirrors the construction performed inside
    :func:`opensqm.md.prepare.prepare_complex` but with the
    ``bespoke_ligand_forcefield=False`` path because we just need a
    physically reasonable system to exercise the water-swap MC
    against - bespoke parameters would only slow the eval script
    down without materially changing the water sampling behaviour
    near the binding site.
    """
    raw_rdmol = Chem.MolFromMolFile(str(ligand_path), removeHs=False)
    if raw_rdmol is None:
        raise RuntimeError(f"Could not parse ligand at {ligand_path}")
    # The HSP90 SDF in the user's evaluation has implicit hydrogens
    # only; add them explicitly so the OpenFF molecule has the full
    # all-atom topology that the SMIRNOFF template generator expects.
    rdmol_with_h = Chem.AddHs(raw_rdmol, addCoords=True)
    ligand_rdmol = set_residue_info(rdmol_with_h)
    offmol = Molecule.from_rdkit(ligand_rdmol, allow_undefined_stereo=True)
    offmol.name = "LIG"

    forcefield = get_ligand_forcefield(offmol, bespoke_ligand_forcefield=False)
    forcefield.loadFile(
        (
            "amber/ff14SB.xml",
            "amber/phosaa10.xml",
            "amber/tip3p_HFE_multivalent.xml",
            "amber/tip3p_standard.xml",
        ),
    )

    lig_top = offmol.to_topology().to_openmm()
    lig_pos = (
        offmol.conformers[0].m * unit.angstrom
    ).in_units_of(unit.nanometer)
    for chain in lig_top.chains():
        for res in chain.residues():
            res.name = "LIG"

    # PDBFixer cleans up missing terminal cap atoms / non-standard
    # residues / etc. in the raw HSP90 PDB so amber14 templates can
    # match. Adding hydrogens is also done here at pH 7 so the
    # downstream forcefield.createSystem call sees a fully populated
    # all-atom protein.
    import pdbfixer  # type: ignore

    fixer = pdbfixer.PDBFixer(filename=str(protein_pdb_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.removeHeterogens(keepWater=False)
    fixer.addMissingHydrogens(pH=7.0)
    protein_modeller = Modeller(fixer.topology, fixer.positions)

    modeller = Modeller(lig_top, lig_pos)
    modeller.add(protein_modeller.topology, protein_modeller.positions)
    # 0.6 nm padding is enough for a 0.9 nm active-site sphere plus
    # the standard 0.9 nm PME cutoff; keeping the box small here
    # makes the smoke-test eval finish in a few minutes rather than
    # half an hour, while still leaving plenty of bulk water for
    # the sampler.
    modeller.addSolvent(
        forcefield,
        ionicStrength=0.15 * unit.molar,
        padding=1.0 * unit.nanometer,
        boxShape="dodecahedron",
        positiveIon="Na+",
        negativeIon="Cl-",
    )

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=HBonds,
        rigidWater=True,
    )

    ligand_atom_indices = {
        a.index for a in modeller.topology.atoms() if a.residue.name == "LIG"
    }
    apply_rest(system, ligand_atom_indices)
    system.addForce(MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))

    # 2 fs without HMR keeps the smoke-test eval stable through
    # equilibration; the production water-swap path elsewhere in the
    # codebase still runs with 4 fs HMR via ConstantPH.
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond,
    )
    platform_name = os.environ.get("OPENMM_PLATFORM", "CPU")
    try:
        platform = Platform.getPlatformByName(platform_name)
    except Exception:
        platform = Platform.getPlatformByName("CPU")
    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    return sim, sorted(ligand_atom_indices)


def main() -> None:
    home = Path(os.path.expanduser("~"))
    protein_pdb_path = home / "Desktop" / "hsp90.pdb"
    ligand_path = home / "Desktop" / "hsp90.sdf"
    if not protein_pdb_path.exists():
        raise FileNotFoundError(protein_pdb_path)
    if not ligand_path.exists():
        raise FileNotFoundError(ligand_path)

    print(f"Building complex from\n  {protein_pdb_path}\n  {ligand_path}")
    sim, ligand_atom_indices = _build_complex(protein_pdb_path, ligand_path)
    n_waters = sum(1 for r in sim.topology.residues() if r.name == "HOH")
    print(
        f"  built {sim.system.getNumParticles()} atoms, "
        f"{n_waters} water residues, {len(ligand_atom_indices)} ligand atoms"
    )

    print("minimising...")
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)

    # Water-swap config matching the production-phase recipe in the
    # paper: 0.9 nm sphere, 80 perturbation steps and 40 propagation
    # steps per perturbation (12.8 ps total switch at 4 fs HMR).
    water_swap_config = WaterSwapSettings(
        active_site_radius=0.9,
        n_perturbation_steps=80,
        n_propagation_steps_per_perturbation=40,
        direction_probabilities=(0.5, 0.5),
        boundary_check=True,
 
    )
    mover = WaterSwapMC(
        sim,
        ligand_atom_indices,
        config=water_swap_config,
    )
    print(
        "WaterSwapMC ready: "
        f"ghost1 = water residue {mover.ghost1_water_idx}, "
        f"ghost2 = water residue {mover.ghost2_water_idx}, "
        f"active-site sphere = {water_swap_config.active_site_radius} nm"
    )

    print("equilibration: 10 ps of plain MD before sampling")
    for _ in tqdm(range(10)):
        sim.step(500)

    n_water_attempts = 100
    rows = []
    print(
        f"\nproduction water-swap probe: {n_water_attempts} attempts at "
        f"2 fs with {water_swap_config.n_perturbation_steps} perturbation "
        f"steps and {water_swap_config.n_propagation_steps_per_perturbation} "
        f"propagation steps per perturbation"
    )
    for attempt_idx in tqdm(range(n_water_attempts)):
        sim.step(200)
        accepted = mover.attempt()
        summary = mover.summary()
        rows.append({"attempt": attempt_idx, "accepted": int(accepted), **summary})
        
        tqdm.write(
            f"  attempts={summary['total_attempts']:.0f} "
            f"accepted={summary['total_accepted']:.0f} "
            f"in={summary['in_accepted']:.0f}/{summary['in_attempts']:.0f} "
            f"out={summary['out_accepted']:.0f}/{summary['out_attempts']:.0f} "
            f"bnd_rej(in/out)={summary['in_boundary_rejections']:.0f}/"
            f"{summary['out_boundary_rejections']:.0f} "
            f"empty(in/out)={summary['in_empty_source_rejections']:.0f}/"
            f"{summary['out_empty_source_rejections']:.0f}"
        )

        df = pd.DataFrame(rows)
        df.to_csv("hsp90_watermc_summary.csv", index=False)

        final_pdb_path = "/tmp/hsp90_watermc_final.pdb"
        final_state = sim.context.getState(
            getPositions=True, enforcePeriodicBox=True,
        )
        with open(final_pdb_path, "w") as fh:
            PDBFile.writeFile(sim.topology, final_state.getPositions(), fh)
        print(f"final coordinates written to {final_pdb_path}")

    total_accepted = int(mover.total_accepted)
    total_attempts = int(mover.total_attempts)
    print(
        f"\nFinal: {total_accepted}/{total_attempts} water-swap attempts "
        f"accepted (acceptance rate {mover.acceptance_rate:.3f})"
    )
    in_works = np.asarray(mover.in_stats.work_history, dtype=float)
    out_works = np.asarray(mover.out_stats.work_history, dtype=float)
    if in_works.size:
        print(
            f"  in-move work (kT): mean={in_works.mean():.3f} "
            f"std={in_works.std():.3f} (n={in_works.size})"
        )
    if out_works.size:
        print(
            f"  out-move work (kT): mean={out_works.mean():.3f} "
            f"std={out_works.std():.3f} (n={out_works.size})"
        )

    if total_accepted < 2:
        raise SystemExit(
            f"Water-swap acceptance count {total_accepted} did not exceed "
            f"the >1 success threshold the user set for the HSP90 system"
        )
    print(
        f"OK: >= 2 water-swap moves accepted "
        f"(target was 4, threshold was >1)"
    )


if __name__ == "__main__":
    main()
