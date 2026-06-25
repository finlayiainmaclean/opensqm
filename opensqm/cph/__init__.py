"""OpenMM-CpH: Constant pH simulation for OpenMM.

This package implements constant pH simulations with OpenMM using discrete
protonation states and hybrid solvent approaches.
"""

__version__ = '1.0.0'

# Import main classes for easy access
# Import utility modules
from . import simulation_config
from .constantph import (
    ConstantPH,
    ResidueState,
    ResidueTitration,
    select_titratable_residue_indices,
    select_titratable_residues,
    select_titratable_residues_by_rdsl,
)
from .ph_remd import ConstantPHRemd
from .reference_energy import ReferenceEnergyFinder

__all__ = [
    'ConstantPH',
    'ConstantPHRemd',
    'ReferenceEnergyFinder',
    'ResidueState',
    'ResidueTitration',
    'WaterSwapMC',
    'WaterSwapSettings',
    'select_titratable_residue_indices',
    'select_titratable_residues',
    'select_titratable_residues_by_rdsl',
    'simulation_config',
]
