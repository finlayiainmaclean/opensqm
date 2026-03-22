# ruff: noqa: E501
"""MOPAC integration: LEWIS/CVB, MOZYME single-point and optimisation, charges, interaction energies."""

from opensqm.mopac.charges import get_mopac_formal_charges
from opensqm.mopac.constants import MOPAC_VDW_RADII, OptMode
from opensqm.mopac.exceptions import MOPACError
from opensqm.mopac.geometry import (
    all_combinations,
    annotate_mopac_formal_charges,
    annotate_mopac_pi_bonds,
    get_mopac_pi_bonds,
    get_rdkit_formal_charges,
    get_rdkit_pi_bonds,
    rdkit_to_mopac,
    write_setpi,
)
from opensqm.mopac.interaction import optimise_complex, run_interaction_energy
from opensqm.mopac.lewis import (
    _prepare_mopac_lewis_bonds_job,
    get_atom_label,
    get_cvb_str,
    get_mopac_bonds,
)
from opensqm.mopac.ligand import get_correct_ligand
from opensqm.mopac.nitro import fix_nitro_groups, strip_mopac_nitro_aux_bonds
from opensqm.mopac.opt_mask import get_opt_mask
from opensqm.mopac.optimize import run_opt_from_rdmol
from opensqm.mopac.parse_output import _parse_bonds, calculate_nonpolar_term
from opensqm.mopac.runner import _run_mopac_input_file, check_mopac_was_success
from opensqm.mopac.singlepoint import run_singlepoint_from_rdmol

__all__ = [
    "MOPAC_VDW_RADII",
    "MOPACError",
    "OptMode",
    "_parse_bonds",
    "_prepare_mopac_lewis_bonds_job",
    "_run_mopac_input_file",
    "all_combinations",
    "annotate_mopac_formal_charges",
    "annotate_mopac_pi_bonds",
    "calculate_nonpolar_term",
    "check_mopac_was_success",
    "fix_nitro_groups",
    "get_atom_label",
    "get_correct_ligand",
    "get_cvb_str",
    "get_mopac_bonds",
    "get_mopac_formal_charges",
    "get_mopac_pi_bonds",
    "get_opt_mask",
    "get_rdkit_formal_charges",
    "get_rdkit_pi_bonds",
    "optimise_complex",
    "rdkit_to_mopac",
    "run_interaction_energy",
    "run_opt_from_rdmol",
    "run_singlepoint_from_rdmol",
    "strip_mopac_nitro_aux_bonds",
    "write_setpi",
]
