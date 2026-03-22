# ruff: noqa: D100, D103

import numpy as np
from rdkit import Chem
from rdsl import select_atom_ids

from opensqm.mopac.constants import OptMode


def get_opt_mask(complex: Chem.Mol, mode: OptMode = "ligand") -> np.ndarray:
    opt_mask = np.zeros(complex.GetNumAtoms(), dtype=bool)

    match mode:
        case "ligand":
            ids = select_atom_ids(complex, "byres (resn LIG)")
            opt_mask[ids] = True

        case "hydrogens":
            ids = select_atom_ids(complex, "elem H within 4 of resn LIG")
            opt_mask[ids] = True

        case "pocket":
            ids = select_atom_ids(complex, "byres (resn LIG) and (sidechain within 4 of resn LIG)")
            opt_mask[ids] = True

    return opt_mask.astype(int)
