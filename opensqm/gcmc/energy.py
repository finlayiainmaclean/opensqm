"""Batched interaction energy of trial waters, for the first stage of a GCMC move."""

from typing import Any

ONE_4PI_EPS0 = 138.93545764438198  # kJ nm / (mol e^2), the value OpenMM 8.5 uses


def min_image(xp: Any, d: Any, box: Any) -> Any:
    """Wrap displacements ``d`` (..., 3) into the minimum image of a reduced triclinic box.

    ``box`` holds OpenMM's box vectors as rows (a along x, b in the xy plane), so
    the c, b, a order below gives the minimum image for any distance below half
    the smallest box width.
    """
    for k in (2, 1, 0):
        d = d - xp.round(d[..., k : k + 1] / box[k, k]) * box[k]
    return d


def water_interaction_energy(
    xp: Any,
    sites: Any,
    site_params: Any,
    xyz: Any,
    params: Any,
    real: Any,
    own_start: Any,
    box: Any,
    cutoff: float,
    rf_dielectric: float,
    max_elements: int = 2**23,
) -> Any:
    """Return the interaction energy (kJ/mol) of each trial water with every real atom.

    Plain 12-6 Lennard-Jones with Lorentz-Berthelot mixing plus reaction-field
    Coulomb, both truncated at ``cutoff``, with the minimum image in ``box``. All
    arrays belong to namespace ``xp`` (numpy or cupy) and the result has the
    dtype of ``xyz``. NaN or inf in the result means an overlap.

    Parameters
    ----------
    xp : module
        numpy or cupy.
    sites : array (T, S, 3)
        Site positions (nm) of the T trial waters.
    site_params : array (S, 3)
        Charge (e), sigma (nm) and epsilon (kJ/mol) of each water site.
    xyz : array (N, 3)
        Positions (nm) of all atoms.
    params : array (N, 3)
        Charge, sigma and epsilon of all atoms.
    real : bool array (N,)
        True for atoms that interact (not ghost waters).
    own_start : int array (T,)
        Index of the first atom of the trial water's own residue, whose S atoms
        are left out (a deletion). Use a large negative value for none (an insertion).
    box : array (3, 3)
        Box vectors (nm) as rows.
    cutoff : float
        Cutoff (nm).
    rf_dielectric : float
        Reaction-field dielectric constant.
    max_elements : int
        Trials are done in chunks of at most this many site-atom-coordinate elements.
    """
    n_trials, n_sites, _ = sites.shape
    k_rf = (rf_dielectric - 1) / ((2 * rf_dielectric + 1) * cutoff**3)
    c_rf = 3 * rf_dielectric / ((2 * rf_dielectric + 1) * cutoff)
    pair_qq, pair_sigma, pair_eps = (
        ONE_4PI_EPS0 * site_params[:, None, 0] * params[None, :, 0],
        0.5 * (site_params[:, None, 1] + params[None, :, 1]),
        xp.sqrt(site_params[:, None, 2] * params[None, :, 2]),
    )
    atom_index = xp.arange(xyz.shape[0])
    chunk = max(1, max_elements // (3 * n_sites * xyz.shape[0]))
    out = []
    # ponytail: every trial against every atom, O(trials x atoms); a cell list around the
    # sphere is the upgrade when the system is large.
    for i in range(0, n_trials, chunk):
        d = min_image(xp, xyz[None, None] - sites[i : i + chunk, :, None], box)
        r2 = (d * d).sum(-1)
        own = atom_index[None, :] - own_start[i : i + chunk, None]
        keep = real[None, :] & ((own < 0) | (own >= n_sites))
        mask = keep[:, None, :] & (r2 < cutoff**2)
        r2 = xp.where(mask, r2, 1.0)
        x = (pair_sigma**2 / r2) ** 3
        e = 4 * pair_eps * x * (x - 1) + pair_qq * (1 / xp.sqrt(r2) + k_rf * r2 - c_rf)
        out.append(xp.where(mask, e, 0.0).sum(axis=(1, 2)))
    return xp.concatenate(out)
