"""Vectorized numpy/scipy kernels for grid-based solvation analysis (GIST).

This module is a pure-Python reimplementation of the computational core that the
original SSTMap package (Haider et al., *J. Chem. Theory Comput.* 2017,
DOI:10.1021/acs.jctc.7b00592) ships as a C extension (``_sstmap_ext``).  Every
routine here mirrors the behaviour of its C counterpart so that the resulting
GIST quantities are numerically equivalent, but the heavy loops are expressed as
``numpy`` array operations instead of compiled C.

The functions are intentionally side-effect free (they take arrays in and return
arrays out) so they can be unit-tested in isolation without any molecular
dynamics input files.
"""

from __future__ import annotations

import numpy as np

# Physical constants, identical to the values hard-coded in the SSTMap C source.
GAS_KCAL = 0.0019872041  # Boltzmann constant in kcal/(mol K)
EULER_MASC = 0.5772156649  # Euler-Mascheroni constant
TWO_PI = 2.0 * np.pi


def detect_orthorhombic(box: np.ndarray, tol: float = 1.0e-4) -> bool:
    """Return ``True`` if a 3x3 unit-cell matrix is (numerically) orthorhombic.

    Parameters
    ----------
    box : numpy.ndarray, shape (3, 3)
        Unit-cell vectors as rows, in Angstrom.
    tol : float, optional
        Absolute tolerance on the off-diagonal elements.

    Returns
    -------
    bool
        Whether the off-diagonal elements are all within ``tol`` of zero.
    """
    off_diag = box - np.diag(np.diagonal(box))
    return bool(np.all(np.abs(off_diag) < tol))


def assign_voxels(
    o_coords: np.ndarray,
    dims: np.ndarray,
    grid_max: np.ndarray,
    origin: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """Assign water oxygen atoms to grid voxels.

    Mirrors ``_sstmap_ext.assign_voxels``: a water is binned only when its oxygen
    falls inside the grid box (and on the non-negative side of the origin).

    Parameters
    ----------
    o_coords : numpy.ndarray, shape (n_water, 3)
        Water oxygen coordinates for a single frame, in Angstrom.
    dims : numpy.ndarray, shape (3,)
        Number of voxels along x, y and z.
    grid_max : numpy.ndarray, shape (3,)
        Upper bound (relative to the origin) past which waters are discarded.
    origin : numpy.ndarray, shape (3,)
        Coordinates of the grid origin (corner), in Angstrom.
    spacing : float
        Grid spacing in Angstrom.

    Returns
    -------
    numpy.ndarray, shape (n_assigned, 2)
        Columns are ``(voxel_id, water_local_index)`` where ``water_local_index``
        indexes into ``o_coords``. The caller maps it back to an atom id.
    """
    translated = o_coords - origin
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])

    inside = (
        (translated[:, 0] >= 0.0)
        & (translated[:, 1] >= 0.0)
        & (translated[:, 2] >= 0.0)
        & (translated[:, 0] <= grid_max[0])
        & (translated[:, 1] <= grid_max[1])
        & (translated[:, 2] <= grid_max[2])
    )

    idx = np.floor(translated / spacing).astype(np.int64)
    inside &= (idx[:, 0] < nx) & (idx[:, 1] < ny) & (idx[:, 2] < nz)
    # floor of a small negative residual can never happen here because inside
    # already requires translated >= 0, but guard anyway.
    inside &= (idx[:, 0] >= 0) & (idx[:, 1] >= 0) & (idx[:, 2] >= 0)

    local = np.nonzero(inside)[0]
    ix, iy, iz = idx[local, 0], idx[local, 1], idx[local, 2]
    voxel_ids = (ix * ny + iy) * nz + iz
    return np.column_stack((voxel_ids, local)).astype(np.int64)


def mic_distance_sq(
    sites: np.ndarray,
    atoms: np.ndarray,
    box: np.ndarray,
    is_ortho: bool,
) -> np.ndarray:
    """Squared minimum-image distances between water sites and all atoms.

    Parameters
    ----------
    sites : numpy.ndarray, shape (n_sites, 3)
        Coordinates of the water sites (O, H, H, ...), in Angstrom.
    atoms : numpy.ndarray, shape (n_atoms, 3)
        Coordinates of every atom in the system, in Angstrom.
    box : numpy.ndarray, shape (3, 3)
        Unit-cell vectors as rows, in Angstrom.
    is_ortho : bool
        Whether the cell is orthorhombic (fast path) or triclinic (27-image
        search, matching the brute-force routine in the SSTMap C source).

    Returns
    -------
    numpy.ndarray, shape (n_sites, n_atoms)
        Squared minimum-image distances.
    """
    n_sites = sites.shape[0]
    if is_ortho:
        lengths = np.array([box[0, 0], box[1, 1], box[2, 2]])
        diff = sites[:, None, :] - atoms[None, :, :]
        diff -= lengths * np.round(diff / lengths)
        return np.einsum("ijk,ijk->ij", diff, diff)

    # Triclinic: wrap both sets into the [0, 1) fractional cell and test the 27
    # neighbouring image cells, keeping the closest. This reproduces
    # ``dist_mic_tric_squared`` from the SSTMap C extension.
    inv = np.linalg.inv(box)
    frac_atoms = atoms @ inv.T
    frac_atoms -= np.floor(frac_atoms)
    frac_sites = sites @ inv.T
    frac_sites -= np.floor(frac_sites)
    real_sites = frac_sites @ box  # wrapped site positions in real space

    shifts = np.array([-1.0, 0.0, 1.0])
    grid = np.array(np.meshgrid(shifts, shifts, shifts, indexing="ij")).reshape(3, -1).T
    # image positions for every atom: (n_atoms, 27, 3)
    images = (frac_atoms[:, None, :] + grid[None, :, :]) @ box

    out = np.empty((n_sites, atoms.shape[0]), dtype=np.float64)
    for s in range(n_sites):
        d = real_sites[s][None, None, :] - images
        out[s] = np.min(np.einsum("ijk,ijk->ij", d, d), axis=1)
    return out


def pairwise_energies(
    dist_sq: np.ndarray,
    acoeff: np.ndarray,
    bcoeff: np.ndarray,
    chg_product: np.ndarray,
    own_columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Lennard-Jones and electrostatic energies for one water against all atoms.

    Reproduces ``_sstmap_ext.calculate_energy``: ``E_lj = A/r^12 - B/r^6`` and
    ``E_elec = q_i q_j / r`` evaluated for each (water-site, atom) pair. The
    interaction of the water with its own atoms is zeroed so it can be summed
    over freely by the caller.

    Parameters
    ----------
    dist_sq : numpy.ndarray, shape (n_sites, n_atoms)
        Squared distances from :func:`mic_distance_sq`.
    acoeff, bcoeff : numpy.ndarray, shape (n_sites, n_atoms)
        Lennard-Jones A and B coefficients (``4 eps sigma^12`` / ``4 eps sigma^6``).
    chg_product : numpy.ndarray, shape (n_sites, n_atoms)
        Product of partial charges (already scaled by 18.2223 so that
        ``q_i q_j / r`` is in kcal/mol).
    own_columns : numpy.ndarray
        Atom indices belonging to the query water itself; these columns are set
        to zero in both returned arrays.

    Returns
    -------
    e_lj, e_elec : numpy.ndarray, shape (n_sites, n_atoms)
        Pairwise Lennard-Jones and electrostatic energies in kcal/mol.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sq = 1.0 / dist_sq
        d6 = inv_sq * inv_sq * inv_sq
        e_lj = acoeff * d6 * d6 - bcoeff * d6
        e_elec = chg_product / np.sqrt(dist_sq)
    e_lj[:, own_columns] = 0.0
    e_elec[:, own_columns] = 0.0
    return e_lj, e_elec


def water_quaternion(owat: np.ndarray, h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Quaternion describing the orientation of a water molecule.

    Direct port of ``GridWaterAnalysis.calculate_euler_angles`` (itself adapted
    from cpptraj's ``Action_GIST``). The lab frame is defined by the x and z
    axes; the quaternion rotates the molecule onto that frame.

    Parameters
    ----------
    owat : numpy.ndarray, shape (3,)
        Oxygen coordinate, in Angstrom.
    h1, h2 : numpy.ndarray, shape (3,)
        Hydrogen coordinates, in Angstrom.

    Returns
    -------
    numpy.ndarray, shape (4,)
        Quaternion ``(w, x, y, z)``.
    """
    xlab = np.array([1.0, 0.0, 0.0])
    zlab = np.array([0.0, 0.0, 1.0])

    h1wat = h1 - owat
    h2wat = h2 - owat
    h1wat /= np.linalg.norm(h1wat)
    h2wat /= np.linalg.norm(h2wat)

    ar1 = np.cross(h1wat, xlab)
    sar = np.copy(ar1)
    ar1 /= np.linalg.norm(ar1)
    dp1 = np.sum(xlab * h1wat)
    theta = np.arccos(dp1)
    sign = np.sum(sar * h1wat)
    theta = theta / 2.0 if sign > 0 else theta / -2.0

    w1 = np.cos(theta)
    sin_theta = np.sin(theta)
    x1 = ar1[0] * sin_theta
    y1 = ar1[1] * sin_theta
    z1 = ar1[2] * sin_theta
    w2, x2, y2, z2 = w1, x1, y1, z1

    h_temp = np.zeros(3)
    h_temp[0] = ((w2 * w2 + x2 * x2) - (y2 * y2 + z2 * z2)) * h1wat[0]
    h_temp[0] = (2 * (x2 * y2 + w2 * z2) * h1wat[1]) + h_temp[0]
    h_temp[0] = (2 * (x2 * z2 - w2 * y2) * h1wat[2]) + h_temp[0]
    h_temp[1] = 2 * (x2 * y2 - w2 * z2) * h1wat[0]
    h_temp[1] = ((w2 * w2 - x2 * x2 + y2 * y2 - z2 * z2) * h1wat[1]) + h_temp[1]
    h_temp[1] = (2 * (y2 * z2 + w2 * x2) * h1wat[2]) + h_temp[1]
    h_temp[2] = 2 * (x2 * z2 + w2 * y2) * h1wat[0]
    h_temp[2] = (2 * (y2 * z2 - w2 * x2) * h1wat[1]) + h_temp[2]
    h_temp[2] = ((w2 * w2 - x2 * x2 - y2 * y2 + z2 * z2) * h1wat[2]) + h_temp[2]

    h_temp2 = np.zeros(3)
    h_temp2[0] = ((w2 * w2 + x2 * x2) - (y2 * y2 + z2 * z2)) * h2wat[0]
    h_temp2[0] = (2 * (x2 * y2 + w2 * z2) * h2wat[1]) + h_temp2[0]
    h_temp2[0] = (2 * (x2 * z2 - w2 * y2) * h2wat[2]) + h_temp2[0]
    h_temp2[1] = 2 * (x2 * y2 - w2 * z2) * h2wat[0]
    h_temp2[1] = ((w2 * w2 - x2 * x2 + y2 * y2 - z2 * z2) * h2wat[1]) + h_temp2[1]
    h_temp2[1] = (2 * (y2 * z2 + w2 * x2) * h2wat[2]) + h_temp2[1]
    h_temp2[2] = 2 * (x2 * z2 + w2 * y2) * h2wat[0]
    h_temp2[2] = (2 * (y2 * z2 - w2 * x2) * h2wat[1]) + h_temp2[2]
    h_temp2[2] = ((w2 * w2 - x2 * x2 - y2 * y2 + z2 * z2) * h2wat[2]) + h_temp2[2]

    ar2 = np.cross(h_temp, h_temp2)
    ar2 /= np.linalg.norm(ar2)
    dp2 = np.sum(ar2 * zlab)
    theta = np.arccos(dp2)
    sar = np.cross(ar2, zlab)
    sign = np.sum(sar * h_temp)
    theta = theta / 2.0 if sign < 0 else theta / -2.0

    w3 = np.cos(theta)
    sin_theta = np.sin(theta)
    x3 = xlab[0] * sin_theta
    y3 = xlab[1] * sin_theta
    z3 = xlab[2] * sin_theta

    w4 = w1 * w3 - x1 * x3 - y1 * y3 - z1 * z3
    x4 = w1 * x3 + x1 * w3 + y1 * z3 - z1 * y3
    y4 = w1 * y3 - x1 * z3 + y1 * w3 + z1 * x3
    z4 = w1 * z3 + x1 * y3 - y1 * x3 + z1 * w3
    return np.array([w4, x4, y4, z4])


def _interior_voxel_mask(dims: np.ndarray) -> np.ndarray:
    """Boolean mask (flattened voxel order) marking non-face voxels.

    Translational/six-D nearest neighbours are searched over the full 3x3x3 voxel
    neighbourhood; only voxels that are not on any outer face of the grid have all
    26 neighbours available. This matches the boundary test in the SSTMap C source.
    """
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    interior = np.zeros((nx, ny, nz), dtype=bool)
    if nx >= 3 and ny >= 3 and nz >= 3:
        interior[1:-1, 1:-1, 1:-1] = True
    return interior.reshape(-1)


def _quat_nn_angle(q_ref: np.ndarray, q_others: np.ndarray) -> np.ndarray:
    """Orientational distance ``2 acos(|q_ref . q_other|)`` for each neighbour."""
    dots = np.abs(q_others @ q_ref)
    np.clip(dots, 0.0, 1.0, out=dots)
    return 2.0 * np.arccos(dots)


def compute_nn_entropy(
    voxeldata: np.ndarray,
    dims: np.ndarray,
    voxel_o_coords: list,
    voxel_quaternions: list,
    num_frames: int,
    voxel_vol: float,
    rho_bulk: float,
    temperature: float,
) -> None:
    """Compute nearest-neighbour translational/orientational/six-D entropies.

    Fills the entropy columns of ``voxeldata`` in place (columns 7-12), exactly
    reproducing ``_sstmap_ext.getNNTrEntropy``:

    * orientational NN is searched among waters in the *same* voxel,
    * translational and six-D NN are searched over the 3x3x3 voxel block (for
      interior voxels) gated by a 3 Angstrom translational cutoff,
    * the Lazaridis nearest-neighbour estimator with the Euler-Mascheroni
      correction converts NN distances into per-water entropies.

    Parameters
    ----------
    voxeldata : numpy.ndarray, shape (n_voxels, 35)
        Voxel data array; modified in place.
    dims : numpy.ndarray, shape (3,)
        Grid dimensions (number of voxels per axis).
    voxel_o_coords : list of array-like
        For each voxel, the flattened oxygen coordinates of every water seen
        there over the whole trajectory.
    voxel_quaternions : list of array-like
        For each voxel, the flattened orientation quaternions of those waters.
    num_frames : int
        Number of processed frames.
    voxel_vol : float
        Voxel volume in Angstrom^3.
    rho_bulk : float
        Reference (bulk) water number density in molecules/Angstrom^3.
    temperature : float
        Temperature in Kelvin.
    """
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    add_x, add_y, add_z = ny * nz, nz, 1
    n_voxels = nx * ny * nz
    pi = np.pi

    o_arrays = [np.asarray(c, dtype=np.float64).reshape(-1, 3) for c in voxel_o_coords]
    q_arrays = [np.asarray(q, dtype=np.float64).reshape(-1, 4) for q in voxel_quaternions]
    interior = _interior_voxel_mask(dims)

    neighbor_offsets = [
        dx * add_x + dy * add_y + dz * add_z
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    for voxel in range(n_voxels):
        nw = voxeldata[voxel, 4]
        if nw < 1:
            continue
        o0 = o_arrays[voxel]
        q0 = q_arrays[voxel]
        n_local = o0.shape[0]

        # --- orientational entropy: nearest neighbour within the same voxel ---
        orient_sum = 0.0
        if n_local > 1:
            for n0 in range(n_local):
                mask = np.arange(n_local) != n0
                rR = _quat_nn_angle(q0[n0], q0[mask])
                rR = rR[rR > 0.0]
                if rR.size:
                    nnr = rR.min()
                    orient_sum += np.log(nnr * nnr * nnr * n_local / (3.0 * TWO_PI))
        voxeldata[voxel, 10] = orient_sum

        # --- candidate set for translational / six-D NN (voxel + neighbours) ---
        if interior[voxel]:
            cand_o = [o0]
            cand_q = [q0]
            for off in neighbor_offsets:
                nb = voxel + off
                if 0 <= nb < n_voxels and o_arrays[nb].shape[0] > 0:
                    cand_o.append(o_arrays[nb])
                    cand_q.append(q_arrays[nb])
            cand_o_all = np.vstack(cand_o)
            cand_q_all = np.vstack(cand_q)
        else:
            cand_o_all = o0
            cand_q_all = q0

        trans_sum = 0.0
        six_sum = 0.0
        for n0 in range(n_local):
            dd = np.einsum("ij,ij->i", cand_o_all - o0[n0], cand_o_all - o0[n0])
            rR = _quat_nn_angle(q0[n0], cand_q_all)
            ds = rR * rR + dd

            dd_pos = dd[dd > 0.0]
            if dd_pos.size == 0:
                continue
            nnd = np.sqrt(dd_pos.min())
            ds_pos = ds[ds > 0.0]
            nns = np.sqrt(ds_pos.min())

            if 0.0 < nnd < 3.0:
                trans_sum += np.log(nnd**3 * num_frames * 4.0 * pi * rho_bulk / 3.0)
                six_sum += np.log(nns**6 * num_frames * pi * rho_bulk / 48.0)
        voxeldata[voxel, 8] = trans_sum
        voxeldata[voxel, 12] = six_sum

        # --- convert the accumulated NN sums into entropies (kcal/mol) ---
        norm = GAS_KCAL * temperature
        dens_factor = nw / (num_frames * voxel_vol)
        if orient_sum != 0.0:
            voxeldata[voxel, 10] = norm * ((orient_sum / nw) + EULER_MASC)
            voxeldata[voxel, 9] = voxeldata[voxel, 10] * dens_factor
        if trans_sum != 0.0:
            voxeldata[voxel, 8] = norm * ((trans_sum / nw) + EULER_MASC)
            voxeldata[voxel, 12] = norm * ((six_sum / nw) + EULER_MASC)
        voxeldata[voxel, 7] = voxeldata[voxel, 8] * dens_factor
        voxeldata[voxel, 11] = voxeldata[voxel, 12] * dens_factor
