from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import itertools
from typing import Any

def henderson_hasselbalch(pH, pKa):
    return 1 / (1 + 10**(pH - pKa))


def compute_populations(df, cph):
    """Return per-residue state populations as a function of pH.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form sampling output with a ``'ph'`` column and one column
        per titratable residue index whose entries are state indices
        into the residue's ``variants`` list.
    cph : ConstantPH
        The constant-pH model that produced ``df``. Used to look up each
        residue's variant names and charges via
        ``cph.titrations[residue_index].reference``.

    Returns
    -------
    dict[int, pandas.DataFrame]
        ``{residue_index: populations_df}`` where ``populations_df`` is
        indexed by pH and has one column per variant whose values are the
        fraction of samples in that variant at that pH. All variants
        appear as columns even if they were never visited at a given pH.
    """
    populations: dict[int, pd.DataFrame] = {}
    for residue_index, titration in cph.titrations.items():
        reference = titration.reference
        charges = reference.charges
        variant_names = reference.variant_names
        fractions = (
            df.groupby('ph')[residue_index]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
        )
        for state in range(len(charges)):
            if state not in fractions.columns:
                fractions[state] = 0.0
        fractions = fractions[list(range(len(charges)))]
        fractions.columns = [
            f"{variant_names[i]}(q={charges[i]:+d})" for i in range(len(charges))
        ]
        populations[residue_index] = fractions
    return populations


def compute_joint_populations(df, cph):
    """Return joint microstate populations as a function of pH.

    For ``N`` titratable residues with ``n_i`` variants each, reports the
    fraction of samples in each of the ``prod(n_i)`` joint protonation
    microstates (one column per combination of per-residue state indices).

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form sampling output with a ``'ph'`` column and one column
        per titratable residue index whose entries are state indices
        into the residue's ``variants`` list.
    cph : ConstantPH
        The constant-pH model that produced ``df``. Used to label each
        joint state by residue name, variant name, and charge.

    Returns
    -------
    pandas.DataFrame
        Indexed by pH; columns are human-readable joint microstate labels;
        values are the fraction of samples in that joint state at that pH.
        All joint microstates appear as columns even if never visited.
    """
    titratable_indices = sorted(cph.titrations.keys())
    if not titratable_indices:
        return pd.DataFrame()

    n_states_per_residue = [
        len(cph.titrations[i].charges) for i in titratable_indices
    ]
    all_joint_states = list(
        itertools.product(*[range(n) for n in n_states_per_residue])
    )

    joint_key = df[titratable_indices].apply(tuple, axis=1)
    fractions = (
        df.assign(_joint=joint_key)
        .groupby('ph')['_joint']
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    for state in all_joint_states:
        if state not in fractions.columns:
            fractions[state] = 0.0
    fractions = fractions[list(all_joint_states)]

    def _joint_label(state_tuple: tuple[int, ...]) -> str:
        parts = []
        for res_idx, state in zip(titratable_indices, state_tuple):
            titration = cph.titrations[res_idx]
            residue = next(
                r for r in cph.explicitTopology.residues() if r.index == res_idx
            )
            name = titration.variant_names[state]
            parts.append(f"{residue.name}.{res_idx}:{name}")
        return " | ".join(parts)

    fractions.columns = [_joint_label(col) for col in fractions.columns]
    return fractions


def flatten_population_vectors(
    per_residue: dict[int, pd.DataFrame],
    joint: pd.DataFrame,
) -> np.ndarray:
    """Stack per-residue and joint population fractions in a fixed order."""
    parts: list[Any] = []
    for residue_index in sorted(per_residue):
        parts.append(
            per_residue[residue_index].to_numpy(dtype=float).ravel()
        )
    if not joint.empty:
        parts.append(joint.to_numpy(dtype=float).ravel())
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def population_distribution_diff(
    per_residue_a: dict[int, pd.DataFrame],
    joint_a: pd.DataFrame,
    per_residue_b: dict[int, pd.DataFrame],
    joint_b: pd.DataFrame,
) -> float:
    """L2 distance between two flattened marginal + joint population vectors."""
    a = flatten_population_vectors(per_residue_a, joint_a)
    b = flatten_population_vectors(per_residue_b, joint_b)
    if a.shape != b.shape:
        raise ValueError(
            f"population vector length mismatch: {a.shape} vs {b.shape}"
        )
    return float(np.linalg.norm(b - a))


def calculate_pkas(df, cph):
    """Fit a macroscopic pKa for each adjacent charge transition per residue.

    Each residue's column in ``df`` records the index of the variant the
    residue currently occupies. ``cph.titrations[i].reference`` provides
    the per-state formal charges (used to bin samples by macroscopic
    charge) and the expected macroscopic pKas used as initial guesses
    for the Henderson-Hasselbalch fit.

    For every adjacent pair of charges the residue can adopt
    (e.g. +1 vs 0 for HIS, then 0 vs -1 if HIN is also enabled), a
    Henderson-Hasselbalch curve is fit to the higher-charge fraction
    restricted to those two charge bins and the macroscopic pKa for that
    charge transition is reported.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form sampling output with a ``'ph'`` column and one column per
        titratable residue index whose entries are state indices into the
        residue's ``variants`` list.
    cph : ConstantPH
        The constant-pH model that produced ``df``.

    Returns
    -------
    dict[int, dict[tuple[int, int], tuple[float, float]]]
        ``{residue_index: {(charge_high, charge_low): (pka, pka_err)}}``.
    """
    pkas: dict[int, dict[tuple[int, int], tuple[float, float]]] = {}
    for residue_index, titration in cph.titrations.items():
        reference = titration.reference
        charges = reference.charges
        reference_pkas = reference.macro_pkas_by_charge_transition
        unique_charges = sorted(set(charges), reverse=True)
        # Map currently-active state index -> macroscopic charge for this residue.
        charge_series = df[residue_index].map(lambda i, ch=charges: ch[i])
        residue_pkas: dict[tuple[int, int], tuple[float, float]] = {}
        for c_high, c_low in zip(unique_charges, unique_charges[1:]):
            mask = charge_series.isin([c_high, c_low])
            if not mask.any():
                continue
            sub = df.loc[mask].copy()
            sub['_charge'] = charge_series[mask]
            frac = (
                sub.groupby('ph')['_charge']
                .apply(lambda x, ch=c_high: float(np.mean(x == ch)))
                .reset_index(name='f_high')
            )
            ref_pka = reference_pkas.get((c_high, c_low), 7.0)
            try:
                popt, pcov = curve_fit(  # type: ignore[misc]
                    henderson_hasselbalch, frac['ph'], frac['f_high'], p0=[ref_pka],
                )
            except Exception:
                continue
            residue_pkas[(c_high, c_low)] = (
                float(popt[0]), float(np.sqrt(np.diag(pcov))[0]),
            )
        pkas[residue_index] = residue_pkas
    return pkas
