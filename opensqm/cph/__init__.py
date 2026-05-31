"""OpenMM-CpH: Constant pH simulation for OpenMM

This package implements constant pH simulations with OpenMM using discrete
protonation states and hybrid solvent approaches.
"""

__version__ = '1.0.0'

# Import main classes for easy access
from .constantph import (
    ConstantPH,
    ResidueState,
    ResidueTitration,
    select_titratable_residues,
)
from .reference_energy import ReferenceEnergyFinder
from opensqm.md.water_swap_mc import WaterSwapSettings, WaterSwapMC

# Import utility modules
from . import simulation_config

__all__ = [
    'ConstantPH',
    'ResidueState',
    'ResidueTitration',
    'ReferenceEnergyFinder',
    'WaterSwapSettings',
    'WaterSwapMC',
    'select_titratable_residues',
    'simulation_config',
]