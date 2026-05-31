import itertools
from typing import Any

import numpy as np
import pandas as pd
from openff.toolkit.topology import Molecule  # type: ignore
from openmm import MonteCarloBarostat, unit
from openmm.app import Modeller, PDBFile
from pydantic import BaseModel, ConfigDict, Field
from pydantic_units import OpenMMQuantity
from rdkit import Chem
from tqdm import tqdm
from unipka import UnipKa

from opensqm.cph.constantph import ConstantPH, select_titratable_residues
from opensqm.cph.pka import (
    calculate_pkas,
    compute_joint_populations,
    compute_populations,
    population_distribution_diff,
)
from opensqm.cph.reference_energy import (
    build_protonation_states,
    build_transitions_tree,
    generate_residue_reference_dict,
)
from opensqm.cph.simulation_config import ConstantpHSettings
from opensqm.cph.trajectory import StateSplitTrajectoryManager
from opensqm.md.equilibrate import EquilibrationSettings, equilibrate
from opensqm.md.prepare import prepare_complex
from opensqm.md.water_swap_mc import WaterSwapSettings
from opensqm.rdkit_utils import set_residue_info
from opensqm.torsion_scanner import autodetect_flip_dihedrals_named
from loguru import logger

def _default_equilibration_config() -> EquilibrationSettings:
    return EquilibrationSettings(
        npt_time=50 * unit.picoseconds,
        warmup_time=10 * unit.picoseconds,
    )


def _default_water_swap_config() -> WaterSwapSettings:
    return WaterSwapSettings(
        active_site_radius=0.9,
        n_perturbation_steps=80,
        n_propagation_steps_per_perturbation=40,
        direction_probabilities=(0.5, 0.5),
        boundary_check=True,
    )


class ConstantpHRunSettings(BaseModel):
    """Settings for a constant-pH MD run."""

    model_config = ConfigDict(frozen=True)
    equilibration_config: EquilibrationSettings = Field(default_factory=_default_equilibration_config)
    cph_config: ConstantpHSettings = Field(default_factory=ConstantpHSettings)
    water_swap_config: WaterSwapSettings = Field(default_factory=_default_water_swap_config)
    residue_distance_cutoff: OpenMMQuantity[unit.angstrom] = 5.0 * unit.angstrom
    production_time: OpenMMQuantity[unit.nanosecond] = 1 * unit.nanosecond
    integrator_step_size: OpenMMQuantity[unit.picosecond] = 0.004 * unit.picoseconds
    ligand_terminal_ring_mc: bool = False
    ligand_protonation: bool = True
    ligand_rest: bool = False
    barostat_pressure: OpenMMQuantity[unit.bar] | None = None
    protonation_swap_interval: OpenMMQuantity[unit.picosecond] = 0.2 * unit.picoseconds
    water_swap_interval: OpenMMQuantity[unit.picosecond] = 0.6 * unit.picoseconds
    reporter_interval: OpenMMQuantity[unit.picosecond] = 1 * unit.picosecond
    max_ligand_protonation_penalty: OpenMMQuantity[unit.kilocalories_per_mole] = 5.0 * unit.kilocalories_per_mole

    
    pH: float = 7.0

    def validate_swap_intervals(self) -> None:
        """Ensure swap intervals are compatible with the integrator step."""
        protonation_ps = self.protonation_swap_interval.value_in_unit(unit.picoseconds)
        water_ps = self.water_swap_interval.value_in_unit(unit.picoseconds)

        if protonation_ps > water_ps:
            msg = "protonation_swap_interval must be <= water_swap_interval"
            raise ValueError(msg)

        ratio = water_ps / protonation_ps
        if abs(ratio - round(ratio)) > 1e-9:
            msg = "water_swap_interval must be an integer multiple of protonation_swap_interval"
            raise ValueError(msg)


def run_cph(
    protein: str,
    ligand: str,
    output: str,
    config: ConstantpHRunSettings = ConstantpHRunSettings(),
    *,
    weights: list[float] | None = None,
) -> dict[str, Any]:
    """Run constant-pH MD for a protein-ligand complex."""

    config.validate_swap_intervals()
    unipka = UnipKa()

    

    ligand_rdmol = set_residue_info(Chem.MolFromMolFile(ligand, removeHs=False))


    if config.ligand_protonation:
        df = unipka.get_distribution(ligand_rdmol).reset_index(drop=True)

        max_ligand_protonation_penalty = config.max_ligand_protonation_penalty.value_in_unit(unit.kilocalories_per_mole)

        df = df[df["relative_ph_adjusted_free_energy"] < max_ligand_protonation_penalty]
        df = df.sort_values(
            ["charge", "relative_ph_adjusted_free_energy"], ascending=[False, True],
        ).reset_index(drop=True)
        print(df)

        state_rdkit_mols = build_protonation_states(list(df["smiles"]))
    else:
        state_rdkit_mols = [ligand_rdmol]

        
    variant_molecules: list[Molecule] = []
    for i, rdmol in enumerate(state_rdkit_mols):
        offmol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
        offmol.name = "LIG" if i == 0 else f"LIG{i}"
        variant_molecules.append(offmol)
    print(variant_molecules)

    if config.ligand_terminal_ring_mc:
        ring_flip_bonds = autodetect_flip_dihedrals_named(state_rdkit_mols[0])
        print("ligand ring_flip_bonds:", ring_flip_bonds)
    else:
        ring_flip_bonds = []

    transitions = build_transitions_tree(
        variant_molecules,
        pka_fn=lambda p, c: unipka.get_macro_pka_from_macrostates(
            acid_macrostate=[p], base_macrostate=[c],
        ),
    )


    main_off = variant_molecules[0]
    protein_pdb = PDBFile(protein)
    protein_modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    omm_top, omm_pos, forcefield = prepare_complex(
        main_off, padding=1.0, protein_modeller=protein_modeller,
    )

    residue_reference_dict = generate_residue_reference_dict(
        config.cph_config,
        ligands=[(variant_molecules, transitions, ring_flip_bonds)],
    )
    print(residue_reference_dict)


    titratable_residue_indices = select_titratable_residues(
        omm_top,
        omm_pos,
        residue_reference_dict,
        ligand_residue_name=main_off.name,
        cutoff=config.residue_distance_cutoff,
    )

    if len(variant_molecules) == 1 and len(ring_flip_bonds)==0:
        # delete the first entry (the ligand)
        # if it is neither titrable or has a ring to flip
        del titratable_residue_indices[0] 
 

    residue_names = np.array([r.name for r in omm_top.residues()])

    titratable_residues_formatted = []
    for i in titratable_residue_indices:
        residue_name = residue_names[i]
        residue_name_formatted = f"{residue_name} ({i})"
        titratable_residues_formatted.append(residue_name_formatted)

    titratable_residues_formatted = ", ".join(titratable_residues_formatted)
    print(titratable_residues_formatted)


    if not titratable_residue_indices:
        raise RuntimeError(
            "No titratable residues were found in the supplied topology"
        )
    


    omm_top, omm_pos = equilibrate(
        omm_top, omm_pos, forcefield, config=config.equilibration_config,
    )

    # pHs = [float(pH) for pH in np.arange(min_pH, max_pH + pH_spacing, pH_spacing)]
    # print("pHs:", pHs)
    pHs = [config.pH]

    cph = ConstantPH(
        topology=omm_top,
        positions=omm_pos,
        pH=pHs,
        config=config.cph_config,
        references=residue_reference_dict,
        ligand_variant_molecules=variant_molecules,
        titratable_residue_indices=titratable_residue_indices,
        water_swap_config=config.water_swap_config,
        rest_residue_names=[main_off.name] if config.ligand_rest else None,
        weights=weights,
    )
    if config.barostat_pressure is not None:
        cph.simulation.system.addForce(MonteCarloBarostat(config.barostat_pressure, 300 * unit.kelvin))
    cph.simulation.context.reinitialize(preserveState=True)

    cur_weights = cph.weights

    logger.info("Optimising weights")
    num_successful_swaps = 0
    equilibration_mc_steps = 10000
    for i in range(equilibration_mc_steps):
        prev_weights = np.array(cur_weights)
        cph.simulation.step(50)  # 0.2ps
        swap = cph.attemptMCStep()
        num_successful_swaps += int(swap)
        cur_weights = np.array(cph.weights)
        diff = np.linalg.norm(cur_weights - prev_weights)

        if i % 250 == 0:
            logger.info(f"Weights difference: {diff}")
            cph.attemptWaterSwap()  # 50ps
        if diff < 1e-1:
            logger.info("Converged")
            break
    else:
        logger.warning("Did not converge")
    logger.info(f"Number of successful swaps: {num_successful_swaps}")
    logger.info(f"Current weights: {cur_weights}")

    cph.reset_stats()

    num_steps = int(config.production_time / config.integrator_step_size)

    results = []
    titratable_indices = list(cph.titrations.keys())
    cur_per_residue_pop: dict[int, pd.DataFrame] | None = None
    cur_joint_pop: pd.DataFrame | None = None


    reporter_interval = int(config.reporter_interval / config.integrator_step_size)

    manager = StateSplitTrajectoryManager(cph, output, report_interval=reporter_interval)
    cph.simulation.reporters.append(manager)

    protonation_swap_steps = int(config.protonation_swap_interval / config.integrator_step_size)
    water_swap_steps = int(config.water_swap_interval / config.integrator_step_size)

    num_batches = num_steps // protonation_swap_steps
    batches_per_water_swap = water_swap_steps // protonation_swap_steps

    

    for i in tqdm(range(num_batches)):
        cph.simulation.step(protonation_swap_steps)
        cph.attemptMCStep()
        result = (
            cph.pH[cph.currentPHIndex],
            *[cph.titrations[index].currentIndex for index in titratable_indices],
        )
        results.append(result)

        if i > 0 and i % batches_per_water_swap == 0:
            cph.attemptWaterSwap()

        if i > 0 and i % 10 == 0: # 0.2ps * 10 = 2ps

            df = pd.DataFrame(results, columns=["ph", *titratable_indices])
            print(calculate_pkas(df, cph))
            new_per_residue_pop = compute_populations(df, cph)
            new_joint_pop = compute_joint_populations(df, cph)
            if cur_per_residue_pop is not None and cur_joint_pop is not None:
                pop_diff = population_distribution_diff(
                    cur_per_residue_pop,
                    cur_joint_pop,
                    new_per_residue_pop,
                    new_joint_pop,
                )
                print(f"population convergence (L2 vs previous checkpoint): {pop_diff:.6f}")

            cur_per_residue_pop = new_per_residue_pop
            cur_joint_pop = new_joint_pop
            print(cph.summary().T)
            print("Water swap stats", cph.waterSwap.summary())
            print("Joint microstate populations:")
            print(new_joint_pop.T)

    manager.close()

    df = pd.DataFrame(results, columns=["ph", *titratable_indices])
    df.to_csv("results.csv", index=False)
    final_pkas = calculate_pkas(df, cph)
    final_populations = compute_populations(df, cph)
    final_joint_populations = compute_joint_populations(df, cph)
    for residue_index, charge_pka_map in final_pkas.items():
        titration = cph.titrations[residue_index]
        residue = next(
            r for r in cph.explicitTopology.residues() if r.index == residue_index
        )
        variant_names = titration.variant_names
        charges = titration.charges
        for (c_high, c_low), (pka, pka_err) in charge_pka_map.items():
            high_names = [n for n, c in zip(variant_names, charges) if c == c_high]
            low_names = [n for n, c in zip(variant_names, charges) if c == c_low]
            print(
                f"  residue {residue.name}.{residue_index} "
                f"{'/'.join(high_names)} ({c_high:+d}) -> "
                f"{'/'.join(low_names)} ({c_low:+d}): "
                f"pKa = {pka:.3f} +/- {pka_err:.3f}"
            )
        print(f"populations for residue {residue.name}.{residue_index}:")
        print(final_populations[residue_index])

    print("\nJoint microstate populations:")
    print(final_joint_populations)

    print("\nFinal per-residue MC stats:")
    print(cph.summary().T)

    print("\nWater swap stats:")
    print(cph.waterSwap.summary())

    return {
        "results": df,
        "pkas": final_pkas,
        "populations": final_populations,
        "joint_populations": final_joint_populations,
        "cph": cph,
    }


if __name__ == "__main__":
    protein_pdb_path = "data/inputs/Wang/Tyk2/4GIH.fixed.pdb"
    ligand_path = "data/inputs/Wang/Tyk2/ejm_42.sdf"

    run_cph(
        protein_pdb_path,
        ligand_path,
        output="/tmp/cph",
        config=ConstantpHRunSettings(),
    )
