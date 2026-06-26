"""Grid Inhomogeneous Solvation Theory (GIST) analysis of MD trajectories.

Pure-Python reimplementation of SSTMap's ``GridWaterAnalysis`` (Haider et al.,
*J. Chem. Theory Comput.* 2017, DOI:10.1021/acs.jctc.7b00592). It maps the
structure and thermodynamics of water onto a 3D grid placed around a region of
interest (typically a ligand binding site), computing per-voxel:

* water number density (``g_O``),
* solute-water and water-water interaction energies (mean-field, no cutoff),
* translational, orientational and six-dimensional entropies via a
  nearest-neighbour estimator,
* first-shell neighbour counts and hydrogen-bond statistics.

The numerically heavy steps are delegated to vectorized kernels in
:mod:`opensqm.sstmap._kernels`, which reproduce the original C extension.

Voxel data layout (``self.voxeldata`` has one row per voxel, 35 columns), using
the column order of the original DX header:

==== ==========================  ==== ==========================
col  quantity                    col  quantity
==== ==========================  ==== ==========================
0    voxel id                    18   E_ww-neighbour-norm
1-3  x, y, z (Angstrom)          19   neighbour-dens
4    population (n_wat)          20   neighbour-norm
5    g_O                         21   f_HB-dens
6    g_H (unused, 0)             22   f_HB-norm
7    dTS_trans-dens              23   N_HB_sw-dens
8    dTS_trans-norm              24   N_HB_sw-norm
9    dTS_orient-dens             25   N_HB_ww-dens
10   dTS_orient-norm             26   N_HB_ww-norm
11   dTS_six-dens                27   N_don_sw-dens
12   dTS_six-norm                28   N_don_sw-norm
13   E_sw-dens                   29   N_acc_sw-dens
14   E_sw-norm                   30   N_acc_sw-norm
15   E_ww-dens                   31   N_don_ww-dens
16   E_ww-norm                   32   N_don_ww-norm
17   E_ww-neighbour-dens         33   N_acc_ww-dens
                                 34   N_acc_ww-norm
==== ==========================  ==== ==========================
"""

from __future__ import annotations

import numpy as np

try:  # mdtraj is optional at import time so the math kernels stay importable.
    import mdtraj as md
except ImportError:  # pragma: no cover
    md = None

from opensqm.sstmap import _kernels
from opensqm.sstmap.water_analysis import WaterAnalysis

N_COLUMNS = 35
NEIGHBOUR_CUTOFF_SQ = 3.5**2  # first-shell O-O cutoff (Angstrom^2)


class GridWaterAnalysis(WaterAnalysis):
    """Grid-based (GIST) water structure and thermodynamics analysis."""

    def __init__(
        self,
        topology_file: str,
        trajectory: str,
        start_frame: int = 0,
        num_frames: int = 0,
        supporting_file: str | None = None,
        ligand_file: str | None = None,
        grid_center: list | np.ndarray | None = None,
        grid_dimensions: list | np.ndarray = (20, 20, 20),
        grid_resolution: list | np.ndarray = (0.5, 0.5, 0.5),
        rho_bulk: float = 0.0334,
        temperature: float = 300.0,
        prefix: str = "test",
    ):
        """Set up a grid-based solvation analysis.

        Parameters
        ----------
        topology_file : str
            System topology (Amber ``.prmtop`` / ``.parm7``).
        trajectory : str
            MD trajectory readable with ``top=topology_file`` (e.g. NetCDF).
        start_frame : int, optional
            First frame to process. Default 0.
        num_frames : int, optional
            Number of frames to process; 0 (default) means "to the end".
        supporting_file : str, optional
            Non-bonded parameter source if the topology is insufficient.
        ligand_file : str, optional
            PDB whose geometric centre defines the grid centre (used only when
            ``grid_center`` is not given).
        grid_center : list, optional
            x, y, z of the grid centre in Angstrom.
        grid_dimensions : list, optional
            Number of voxels along each axis. Default (20, 20, 20).
        grid_resolution : list, optional
            Grid spacing in Angstrom (isotropic). Default (0.5, 0.5, 0.5).
        rho_bulk : float, optional
            Reference bulk water density (molecules/Angstrom^3). Default 0.0334.
        temperature : float, optional
            Temperature in Kelvin used for entropy. Default 300.
        prefix : str, optional
            Prefix for output file names. Default "test".
        """
        if md is None:  # pragma: no cover
            raise ImportError("mdtraj is required for GridWaterAnalysis.")
        print("Initializing ...")
        self.start_frame = int(start_frame)
        self.num_frames = int(num_frames)
        self.rho_bulk = float(rho_bulk)
        self.temperature = float(temperature)
        self.prefix = prefix
        super().__init__(topology_file, trajectory, supporting_file)

        self.grid_dims = np.asarray(grid_dimensions, dtype=np.int64)
        self.resolution = float(grid_resolution[0])
        self.voxel_vol = self.resolution**3

        if ligand_file is None and grid_center is None:
            raise ValueError(
                "Provide either grid_center (x, y, z) or a ligand_file whose "
                "centre is used as the grid centre."
            )
        if grid_center is None:
            lig = md.load_pdb(ligand_file, no_boxchk=True)
            grid_center = lig.xyz[0].mean(axis=0) * 10.0  # nm -> Angstrom

        self._initialize_grid(grid_center)
        self.voxeldata = self._initialize_voxel_data()
        # Per-voxel accumulators for entropy (filled only when entropy is on).
        n_vox = self.voxeldata.shape[0]
        self.voxel_O_coords: list[list[float]] = [[] for _ in range(n_vox)]
        self.voxel_quarts: list[list[float]] = [[] for _ in range(n_vox)]

    # -- grid construction -------------------------------------------------

    def _initialize_grid(self, center: list | np.ndarray) -> None:
        """Set grid centre, origin, extent and per-axis voxel counts."""
        self.center = np.asarray(center, dtype=np.float64)
        self.dims = self.grid_dims.astype(np.int64)
        self.spacing = np.array(
            [self.resolution, self.resolution, self.resolution], dtype=np.float64
        )
        # Waters are kept if (oxygen - origin) <= grid_max along every axis.
        self.grid_max = self.dims * self.spacing + 1.5
        origin = self.center - (0.5 * self.dims * self.spacing)
        self.origin = np.around(origin, decimals=3)
        self.grid = np.zeros(self.dims, dtype=np.int64)

    def _initialize_voxel_data(self) -> np.ndarray:
        """Allocate the voxel data array and fill voxel centre coordinates."""
        n_vox = int(np.prod(self.dims))
        voxeldata = np.zeros((n_vox, N_COLUMNS), dtype=np.float64)
        ix, iy, iz = np.unravel_index(np.arange(n_vox), tuple(int(d) for d in self.dims))
        idx = np.column_stack((ix, iy, iz)).astype(np.float64)
        centers = idx * self.spacing + self.origin + 0.5 * self.spacing
        voxeldata[:, 0] = np.arange(n_vox)
        voxeldata[:, 1:4] = centers
        return voxeldata

    # -- per-frame processing ---------------------------------------------

    def _process_frame(self, trj, energy: bool, hbonds: bool, entropy: bool) -> None:
        """Accumulate GIST quantities for a single trajectory frame."""
        coords = trj.xyz[0].astype(np.float64) * 10.0  # nm -> Angstrom
        box = trj.unitcell_vectors[0].astype(np.float64) * 10.0
        is_ortho = _kernels.detect_orthorhombic(box)

        o_coords = coords[self.wat_oxygen_atom_ids]
        waters = _kernels.assign_voxels(
            o_coords, self.dims, self.grid_max, self.origin, self.resolution
        )

        site_offsets = np.arange(self.water_sites)
        for voxel_id, local in waters:
            wat_o = int(self.wat_oxygen_atom_ids[local])
            self.voxeldata[voxel_id, 4] += 1

            if energy or hbonds:
                own_columns = wat_o + site_offsets
                sites_xyz = coords[own_columns]
                dist_sq = _kernels.mic_distance_sq(sites_xyz, coords, box, is_ortho)
                e_lj, e_elec = _kernels.pairwise_energies(
                    dist_sq, self.acoeff, self.bcoeff, self.chg_product, own_columns
                )

                # First-shell water neighbours (O-O within 3.5 A, excluding self).
                o_dist_sq = dist_sq[0, self.wat_oxygen_atom_ids]
                nbr_mask = (o_dist_sq < NEIGHBOUR_CUTOFF_SQ) & (o_dist_sq > 0.0)
                wat_nbrs = self.wat_oxygen_atom_ids[nbr_mask]
                self.voxeldata[voxel_id, 19] += wat_nbrs.shape[0]

                if self.non_water_atom_ids.shape[0] != 0:
                    self.voxeldata[voxel_id, 13] += (
                        e_lj[:, self.non_water_atom_ids].sum()
                        + e_elec[:, self.non_water_atom_ids].sum()
                    )
                # Water-water: every other water atom (own columns already zeroed).
                self.voxeldata[voxel_id, 15] += (
                    e_lj[:, self.wat_atom_ids].sum() + e_elec[:, self.wat_atom_ids].sum()
                )
                if wat_nbrs.shape[0] > 0:
                    nbr_cols = (wat_nbrs[:, None] + site_offsets).ravel()
                    self.voxeldata[voxel_id, 17] += (
                        e_lj[:, nbr_cols].sum() + e_elec[:, nbr_cols].sum()
                    )

                if hbonds:
                    self._accumulate_hbonds(trj, voxel_id, wat_o, wat_nbrs, dist_sq)

            if entropy:
                h1 = coords[wat_o + 1]
                h2 = coords[wat_o + 2]
                quart = _kernels.water_quaternion(coords[wat_o], h1, h2)
                self.voxel_O_coords[voxel_id].extend(coords[wat_o].tolist())
                self.voxel_quarts[voxel_id].extend(quart.tolist())

    def _accumulate_hbonds(self, trj, voxel_id, wat_o, wat_nbrs, dist_sq) -> None:
        """Accumulate water-water and solute-water hydrogen-bond statistics."""
        prot_dist_sq = dist_sq[0, self.prot_atom_ids]
        prot_nbrs_all = self.prot_atom_ids[prot_dist_sq <= NEIGHBOUR_CUTOFF_SQ]
        prot_nbrs_hb = prot_nbrs_all[self.prot_hb_types[prot_nbrs_all] != 0]

        if wat_nbrs.shape[0] > 0:
            hb_ww = self.calculate_hydrogen_bonds(trj, wat_o, wat_nbrs)
            acc_ww = hb_ww[hb_ww[:, 0] == wat_o].shape[0]
            don_ww = hb_ww.shape[0] - acc_ww
            self.voxeldata[voxel_id, 25] += hb_ww.shape[0]
            self.voxeldata[voxel_id, 31] += don_ww
            self.voxeldata[voxel_id, 33] += acc_ww
            if hb_ww.shape[0] != 0:
                self.voxeldata[voxel_id, 21] += wat_nbrs.shape[0] / hb_ww.shape[0]

        if prot_nbrs_hb.shape[0] > 0:
            hb_sw = self.calculate_hydrogen_bonds(trj, wat_o, prot_nbrs_hb, water_water=False)
            acc_sw = hb_sw[hb_sw[:, 0] == wat_o].shape[0]
            don_sw = hb_sw.shape[0] - acc_sw
            self.voxeldata[voxel_id, 23] += hb_sw.shape[0]
            self.voxeldata[voxel_id, 27] += don_sw
            self.voxeldata[voxel_id, 29] += acc_sw

    # -- driver ------------------------------------------------------------

    def calculate_grid_quantities(
        self, energy: bool = True, entropy: bool = True, hbonds: bool = True
    ) -> None:
        """Run the GIST calculation over the trajectory.

        Iterates over frames accumulating per-voxel quantities, normalizes the
        energy / structural columns, then evaluates the nearest-neighbour
        entropies. Results are written into ``self.voxeldata``.

        Parameters
        ----------
        energy : bool, optional
            Compute solute-water and water-water energies. Default True.
        entropy : bool, optional
            Compute translational/orientational/six-D entropies. Default True.
        hbonds : bool, optional
            Compute hydrogen-bond statistics. Default True.
        """
        processed = 0
        for chunk in md.iterload(
            self.trajectory, top=self.topology_file, chunk=1, skip=self.start_frame
        ):
            if self.num_frames and processed >= self.num_frames:
                break
            self._process_frame(chunk, energy, hbonds, entropy)
            processed += 1
        if processed == 0:
            raise ValueError("No frames were processed; check the trajectory and start_frame.")
        self.num_frames = processed

        self._normalize(energy or hbonds)
        if entropy:
            _kernels.compute_nn_entropy(
                self.voxeldata,
                self.dims,
                self.voxel_O_coords,
                self.voxel_quarts,
                self.num_frames,
                self.voxel_vol,
                self.rho_bulk,
                self.temperature,
            )
        # g_O density is always meaningful; fill it whether or not entropy ran.
        nwat = self.voxeldata[:, 4]
        self.voxeldata[:, 5] = nwat / (self.num_frames * self.voxel_vol * self.rho_bulk)

    def _normalize(self, did_energy: bool) -> None:
        """Normalize energy/structure columns to density and per-water forms."""
        if not did_energy:
            return
        vd = self.voxeldata
        nwat = vd[:, 4]
        occ = nwat > 1.0
        denom = self.num_frames * self.voxel_vol

        # Energies carry an extra factor of two (GIST convention).
        vd[occ, 14] = vd[occ, 13] / (nwat[occ] * 2.0)
        vd[occ, 13] /= denom * 2.0
        vd[occ, 16] = vd[occ, 15] / (nwat[occ] * 2.0)
        vd[occ, 15] /= denom * 2.0

        nbr = occ & (vd[:, 19] > 0.0)
        vd[nbr, 18] = vd[nbr, 17] / (vd[nbr, 19] * 2.0)
        vd[nbr, 17] /= denom * vd[nbr, 19] * 2.0

        for i in range(19, 35, 2):
            vd[occ, i + 1] = vd[occ, i] / nwat[occ]
            vd[occ, i] /= denom

        # Voxels with <= 1 water cannot give meaningful energies/structure.
        sparse = ~occ
        vd[sparse, 13] = 0.0
        vd[sparse, 15] = 0.0
        vd[sparse, 17] = 0.0
        vd[sparse, 19:35] = 0.0

    # -- output ------------------------------------------------------------

    GIST_COLUMNS = (
        "voxel x y z nwat gO gH dTStrans-dens dTStrans-norm dTSorient-dens dTSorient-norm "
        "dTSsix-dens dTSsix-norm Esw-dens Esw-norm Eww-dens Eww-norm Eww-nbr-dens Eww-nbr-norm "
        "neighbor-dens neighbor-norm fHB-dens fHB-norm Nhbsw-dens Nhbsw-norm Nhbww-dens "
        "Nhbww-norm Ndonsw-dens Ndonsw-norm Naccsw-dens Naccsw-norm Ndonww-dens Ndonww-norm "
        "Naccww-dens Naccww-norm"
    )

    def write_data(self, prefix: str | None = None) -> str:
        """Write a whitespace-delimited table of per-voxel GIST quantities.

        Parameters
        ----------
        prefix : str, optional
            Output file prefix; defaults to the prefix given at construction.

        Returns
        -------
        str
            Path of the written ``<prefix>_gist_data.txt`` file.
        """
        prefix = prefix or self.prefix
        path = f"{prefix}_gist_data.txt"
        with open(path, "w") as f:
            f.write(self.GIST_COLUMNS + "\n")
            for row in self.voxeldata:
                fields = [f"{row[0]:.0f}", f"{row[1]:.3f}", f"{row[2]:.3f}", f"{row[3]:.3f}",
                          f"{row[4]:.0f}"]
                fields.extend(f"{row[c]:.6f}" for c in range(5, N_COLUMNS))
                f.write(" ".join(fields) + "\n")
        return path

    def generate_dx_files(self, prefix: str | None = None) -> list[str]:
        """Write each density-form GIST quantity as an OpenDX grid file.

        Parameters
        ----------
        prefix : str, optional
            Output file prefix; defaults to the prefix given at construction.

        Returns
        -------
        list of str
            Paths of the written ``.dx`` files.
        """
        prefix = prefix or self.prefix
        keys = self.GIST_COLUMNS.split()
        header = (
            f"object 1 class gridpositions counts {self.dims[0]} {self.dims[1]} {self.dims[2]}\n"
            f"origin {self.origin[0]:.3f} {self.origin[1]:.3f} {self.origin[2]:.3f}\n"
            f"delta {self.spacing[0]:.1f} 0 0\n"
            f"delta 0 {self.spacing[1]:.1f} 0\n"
            f"delta 0 0 {self.spacing[2]:.1f}\n"
            f"object 2 class gridconnections counts {self.dims[0]} {self.dims[1]} {self.dims[2]}\n"
            f"object 3 class array type double rank 0 items {self.voxeldata.shape[0]} "
            "data follows\n"
        )
        written = []
        for col, title in enumerate(keys):
            # density-form quantities live in odd columns >= 5 (skip gH).
            if col > 4 and col % 2 == 1 and title != "gH":
                path = f"{prefix}_{title}.dx"
                with open(path, "w") as f:
                    f.write(header)
                    values = self.voxeldata[:, col]
                    for k, val in enumerate(values, start=1):
                        f.write(f"{val:g} ")
                        if k % 3 == 0:
                            f.write("\n")
                    if len(values) % 3 != 0:
                        f.write("\n")
                written.append(path)
        return written

    def print_system_summary(self) -> None:
        """Print a short summary of the molecular system and grid."""
        print("System information:")
        print(f"\tTopology: {self.topology_file}")
        print(f"\tTrajectory: {self.trajectory}")
        print(
            f"\tFrames: {self.num_frames}, Atoms: {self.all_atom_ids.shape[0]}, "
            f"Waters: {self.wat_oxygen_atom_ids.shape[0]}, "
            f"Solute atoms: {self.non_water_atom_ids.shape[0]}"
        )
        print("Grid information:")
        print(f"\tCentre: {self.center[0]:.3f} {self.center[1]:.3f} {self.center[2]:.3f}")
        print(f"\tDimensions: {self.dims[0]} {self.dims[1]} {self.dims[2]}")
        print(f"\tSpacing: {self.spacing[0]:.3f} Angstrom")
