"""Types."""

from dataclasses import dataclass

from rdkit import Chem


@dataclass
class Complex:
    """A protein-ligand complex."""

    protein: Chem.Mol
    ligand: Chem.Mol
