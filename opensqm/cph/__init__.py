"""OpenMM-CpH: Constant pH simulation for OpenMM

This package implements constant pH simulations with OpenMM using discrete
protonation states and hybrid solvent approaches.
"""

__version__ = '1.0.0'

# Import main classes for easy access
from .constantph import ConstantPH, ResidueState, ResidueTitration
from .reference_energy import ReferenceEnergyFinder

# Import utility modules
from . import utils
from . import simulation_config

__all__ = [
    'ConstantPH',
    'ResidueState', 
    'ResidueTitration',
    'ReferenceEnergyFinder',
    'utils',
    'simulation_config',
]