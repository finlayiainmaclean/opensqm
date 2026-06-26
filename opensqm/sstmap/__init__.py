"""Pure-Python implementation of SSTMap-style water analysis.

This package reimplements the grid-based (GIST) water structure and
thermodynamics analysis of SSTMap (Haider, Cruz, Ramsey, Gilson and Kurtzman,
*J. Chem. Theory Comput.* 2017, DOI:10.1021/acs.jctc.7b00592) entirely in Python
(numpy / scipy / mdtraj / parmed), with no compiled extensions.

Example
-------
Run a grid-based calculation with all quantities::

    from opensqm import sstmap as sm

    gist = sm.GridWaterAnalysis(
        "casp3.prmtop",
        "casp3.netcdf",
        ligand_file="casp3_ligand.pdb",
        grid_dimensions=[40, 40, 40],
        prefix="casp3",
    )
    gist.calculate_grid_quantities()
    gist.write_data()
    gist.generate_dx_files()
"""

from opensqm.sstmap.grid_water_analysis import GridWaterAnalysis
from opensqm.sstmap.water_analysis import WaterAnalysis

__all__ = ["GridWaterAnalysis", "WaterAnalysis"]
