"""pKa fitting and titration population/correlation analysis for constant-pH runs."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from opensqm.cph.constantph import residue_label, residue_label_slug

if TYPE_CHECKING:
    from opensqm.cph.constantph import ConstantPH


def henderson_hasselbalch(ph: float | np.ndarray, pka: float) -> float | np.ndarray:
    """Return the high-charge fraction from the Henderson-Hasselbalch relation."""
    return 1 / (1 + 10 ** (ph - pka))


def compute_populations(df: pd.DataFrame, cph: ConstantPH) -> dict[int, pd.DataFrame]:
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
            df.groupby("ph")[residue_index].value_counts(normalize=True).unstack(fill_value=0.0)
        )
        for state in range(len(charges)):
            if state not in fractions.columns:
                fractions[state] = 0.0
        fractions = fractions[list(range(len(charges)))]
        fractions.columns = [f"{variant_names[i]}(q={charges[i]:+d})" for i in range(len(charges))]
        populations[residue_index] = fractions
    return populations


def joint_microstate_label(cph: ConstantPH, state_tuple: Sequence[int]) -> str:
    """Human-readable label for a joint protonation microstate.

    ``state_tuple`` holds one variant index per titratable residue, ordered by
    ``sorted(cph.titrations)`` - the same order used for the columns of
    :func:`compute_joint_populations` and for the trajectory manager's
    ``system_state`` signature. Each residue contributes
    ``"<residue_label>:<variant_name>"`` and the parts are joined by " | ", using
    the residue's *original* structure numbering (see :func:`residue_label`) so
    the label is stable across the run and can be matched against the
    joint-population table.
    """
    titratable_indices = sorted(cph.titrations.keys())
    residues_by_index = {r.index: r for r in cph.explicitTopology.residues()}
    parts = []
    for res_idx, state in zip(titratable_indices, state_tuple, strict=False):
        titration = cph.titrations[res_idx]
        parts.append(
            f"{residue_label(residues_by_index[res_idx])}:{titration.variant_names[state]}"
        )
    return " | ".join(parts)


def compute_joint_populations(df: pd.DataFrame, cph: ConstantPH) -> pd.DataFrame:
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

    n_states_per_residue = [len(cph.titrations[i].charges) for i in titratable_indices]
    all_joint_states = list(itertools.product(*[range(n) for n in n_states_per_residue]))

    joint_key = df[titratable_indices].apply(tuple, axis=1)
    fractions = (
        df.assign(_joint=joint_key)
        .groupby("ph")["_joint"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    for state in all_joint_states:
        if state not in fractions.columns:
            fractions[state] = 0.0
    fractions = fractions[list(all_joint_states)]

    fractions.columns = [joint_microstate_label(cph, col) for col in fractions.columns]
    return fractions


def compute_residue_correlations(
    joint_populations: pd.DataFrame,
    cph: ConstantPH,
) -> pd.DataFrame:
    """Pearson correlation of formal charges between residue pairs at each pH.

    Uses the joint microstate probabilities from
    :func:`compute_joint_populations` to compute, at each pH, the linear
    correlation between the formal charges of every titratable residue pair.

    Parameters
    ----------
    joint_populations : pandas.DataFrame
        Output of :func:`compute_joint_populations`, indexed by pH.
    cph : ConstantPH
        The constant-pH model that produced ``joint_populations``.

    Returns
    -------
    pandas.DataFrame
        Long format with columns ``ph``, ``residue_i_index``,
        ``residue_j_index``, ``residue_i``, ``residue_j``, and
        ``correlation``. Only unique residue pairs (``i < j``) are included.
    """
    titratable_indices = sorted(cph.titrations.keys())
    if len(titratable_indices) < 2 or joint_populations.empty:
        return pd.DataFrame(
            columns=[
                "ph",
                "residue_i_index",
                "residue_j_index",
                "residue_i",
                "residue_j",
                "correlation",
            ]
        )

    n_states_per_residue = [len(cph.titrations[i].charges) for i in titratable_indices]
    all_joint_states = list(itertools.product(*[range(n) for n in n_states_per_residue]))
    charges_by_residue = {
        res_idx: np.asarray(cph.titrations[res_idx].charges, dtype=float)
        for res_idx in titratable_indices
    }
    residue_labels = {
        res_idx: residue_label(
            next(r for r in cph.explicitTopology.residues() if r.index == res_idx)
        )
        for res_idx in titratable_indices
    }

    rows: list[dict[str, Any]] = []
    for ph, probs in joint_populations.iterrows():
        p = probs.to_numpy(dtype=float)
        n_res = len(titratable_indices)
        means = np.zeros(n_res)
        second_moments = np.zeros(n_res)
        cross = np.zeros((n_res, n_res))

        for prob, state in zip(p, all_joint_states, strict=False):
            charge_vec = np.array(
                [
                    charges_by_residue[res_idx][state[res_pos]]
                    for res_pos, res_idx in enumerate(titratable_indices)
                ]
            )
            means += prob * charge_vec
            second_moments += prob * charge_vec**2
            cross += prob * np.outer(charge_vec, charge_vec)

        variances = second_moments - means**2
        cov = cross - np.outer(means, means)

        for i in range(n_res):
            for j in range(i + 1, n_res):
                if variances[i] > 0.0 and variances[j] > 0.0:
                    corr = cov[i, j] / np.sqrt(variances[i] * variances[j])
                else:
                    corr = np.nan
                res_i = titratable_indices[i]
                res_j = titratable_indices[j]
                rows.append(
                    {
                        "ph": ph,
                        "residue_i_index": res_i,
                        "residue_j_index": res_j,
                        "residue_i": residue_labels[res_i],
                        "residue_j": residue_labels[res_j],
                        "correlation": float(corr),
                    }
                )

    return pd.DataFrame(rows).dropna()


def calculate_pkas(
    df: pd.DataFrame, cph: ConstantPH
) -> dict[int, dict[tuple[int, int], tuple[float, float]]]:
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
        for c_high, c_low in itertools.pairwise(unique_charges):
            mask = charge_series.isin([c_high, c_low])
            if not mask.any():
                continue
            sub = df.loc[mask].copy()
            sub["_charge"] = charge_series[mask]
            if set(sub["_charge"].unique()) != {c_high, c_low}:
                # e.g. a serial replica that never visits one charge state.
                continue
            frac = (
                sub.groupby("ph")["_charge"]
                .apply(lambda x, ch=c_high: float(np.mean(x == ch)))
                .reset_index(name="f_high")
            )
            ref_pka = reference_pkas.get((c_high, c_low), 7.0)
            try:
                popt, pcov = curve_fit(  # type: ignore[misc]
                    henderson_hasselbalch,
                    frac["ph"],
                    frac["f_high"],
                    p0=[ref_pka],
                )
            except Exception:
                continue
            residue_pkas[(c_high, c_low)] = (
                float(popt[0]),
                float(np.sqrt(np.diag(pcov))[0]),
            )
        pkas[residue_index] = residue_pkas
    return pkas


def _charge_transition_label(
    cph: ConstantPH,
    residue_index: int,
    c_high: int,
    c_low: int,
) -> str:
    titration = cph.titrations[residue_index]
    residue = next(r for r in cph.explicitTopology.residues() if r.index == residue_index)
    variant_names = titration.variant_names
    charges = titration.charges
    high_names = [n for n, c in zip(variant_names, charges, strict=False) if c == c_high]
    low_names = [n for n, c in zip(variant_names, charges, strict=False) if c == c_low]
    return (
        f"{residue_label(residue)} "
        f"{'/'.join(high_names)} ({c_high:+d}) -> "
        f"{'/'.join(low_names)} ({c_low:+d})"
    )


def _timeseries_sample_indices(n_samples: int, max_points: int = 200) -> np.ndarray:
    if n_samples <= max_points:
        return np.arange(1, n_samples + 1, dtype=int)
    return np.unique(np.linspace(1, n_samples, max_points, dtype=int))


def compute_pka_timeseries(
    df: pd.DataFrame,
    cph: ConstantPH,
    *,
    sample_interval_ns: float,
    max_points: int = 200,
) -> pd.DataFrame:
    """Track cumulative pKa estimates as production samples accrue.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form sampling output with a ``'ph'`` column and one column per
        titratable residue index.
    cph : ConstantPH
        The constant-pH model that produced ``df``.
    sample_interval_ns : float
        Elapsed production time between successive rows in ``df``.
    max_points : int, optional
        Maximum number of cumulative prefixes to evaluate. When ``df`` has
        more rows than this, evenly spaced prefixes are used.

    Returns
    -------
    pandas.DataFrame
        Long format with columns ``time_ns``, ``n_samples``,
        ``residue_index``, ``charge_high``, ``charge_low``, ``label``,
        ``pka``, and ``pka_err``.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "time_ns",
                "n_samples",
                "residue_index",
                "charge_high",
                "charge_low",
                "label",
                "pka",
                "pka_err",
            ]
        )

    rows: list[dict[str, Any]] = []
    for n_samples in _timeseries_sample_indices(len(df), max_points=max_points):
        prefix = df.iloc[:n_samples]
        time_ns = n_samples * sample_interval_ns
        pkas = calculate_pkas(prefix, cph)
        for residue_index, charge_pka_map in pkas.items():
            for (c_high, c_low), (pka, pka_err) in charge_pka_map.items():
                rows.append(
                    {
                        "time_ns": time_ns,
                        "n_samples": n_samples,
                        "residue_index": residue_index,
                        "charge_high": c_high,
                        "charge_low": c_low,
                        "label": _charge_transition_label(cph, residue_index, c_high, c_low),
                        "pka": pka,
                        "pka_err": pka_err,
                    }
                )

    return pd.DataFrame(rows)


def compute_pka_timeseries_from_replicas(
    replica_dfs: Sequence[pd.DataFrame],
    cph: ConstantPH,
    *,
    sample_interval_ns: float,
    max_points: int = 200,
) -> pd.DataFrame:
    """Track cumulative pooled pKa estimates across replicas at each time point.

    At each prefix length ``n``, the first ``n`` samples from every replica are
    concatenated before fitting. ``time_ns`` therefore reflects per-replica
    production time (not the length of the vertically stacked replica table).
    """
    if not replica_dfs:
        return compute_pka_timeseries(pd.DataFrame(), cph, sample_interval_ns=sample_interval_ns)

    n_per_replica = min(len(df) for df in replica_dfs)
    rows: list[dict[str, Any]] = []
    for n_samples in _timeseries_sample_indices(n_per_replica, max_points=max_points):
        pooled = pd.concat(
            (df.iloc[:n_samples] for df in replica_dfs),
            ignore_index=True,
        )
        time_ns = n_samples * sample_interval_ns
        pkas = calculate_pkas(pooled, cph)
        for residue_index, charge_pka_map in pkas.items():
            for (c_high, c_low), (pka, pka_err) in charge_pka_map.items():
                rows.append(
                    {
                        "time_ns": time_ns,
                        "n_samples": n_samples,
                        "residue_index": residue_index,
                        "charge_high": c_high,
                        "charge_low": c_low,
                        "label": _charge_transition_label(cph, residue_index, c_high, c_low),
                        "pka": pka,
                        "pka_err": pka_err,
                    }
                )

    return pd.DataFrame(rows)


def compute_complement_pka_timeseries(
    replica_dfs: Sequence[pd.DataFrame],
    target_replica: int,
    cph: ConstantPH,
    *,
    sample_interval_ns: float,
    max_points: int = 200,
) -> pd.DataFrame:
    """Cumulative pKa with one replica prefix plus full complementary replicas.

    Used when a serial replica never visits both charge states alone (e.g. one
    replica holds ASH, another holds ASP) but the pooled fit is well-defined.
    """
    if not replica_dfs or not (0 <= target_replica < len(replica_dfs)):
        return compute_pka_timeseries(pd.DataFrame(), cph, sample_interval_ns=sample_interval_ns)

    n_per_replica = min(len(df) for df in replica_dfs)
    rows: list[dict[str, Any]] = []
    for n_samples in _timeseries_sample_indices(n_per_replica, max_points=max_points):
        parts = [
            df.iloc[:n_samples] if replica_i == target_replica else df
            for replica_i, df in enumerate(replica_dfs)
        ]
        pooled = pd.concat(parts, ignore_index=True)
        time_ns = n_samples * sample_interval_ns
        pkas = calculate_pkas(pooled, cph)
        for residue_index, charge_pka_map in pkas.items():
            for (c_high, c_low), (pka, pka_err) in charge_pka_map.items():
                rows.append(
                    {
                        "time_ns": time_ns,
                        "n_samples": n_samples,
                        "residue_index": residue_index,
                        "charge_high": c_high,
                        "charge_low": c_low,
                        "label": _charge_transition_label(cph, residue_index, c_high, c_low),
                        "pka": pka,
                        "pka_err": pka_err,
                    }
                )

    return pd.DataFrame(rows)


def _timeseries_transition_keys(df: pd.DataFrame) -> set[tuple[int, int, int]]:
    if df.empty:
        return set()
    return {
        (int(r.residue_index), int(r.charge_high), int(r.charge_low))
        for r in df.itertuples(index=False)
    }


def build_replica_overlay_timeseries(
    replica_dfs: Sequence[pd.DataFrame],
    per_replica_timeseries: Sequence[tuple[str, pd.DataFrame]],
    cph: ConstantPH,
    *,
    sample_interval_ns: float,
) -> list[tuple[str, pd.DataFrame]]:
    """Augment per-replica overlays with complement fits where needed."""
    overlays: list[tuple[str, pd.DataFrame]] = []
    for replica_i, (label, timeseries) in enumerate(per_replica_timeseries):
        complement = compute_complement_pka_timeseries(
            replica_dfs,
            replica_i,
            cph,
            sample_interval_ns=sample_interval_ns,
        )
        own_keys = _timeseries_transition_keys(timeseries)
        if complement.empty:
            merged = timeseries
        elif not own_keys:
            merged = complement
        else:
            complement_only = complement[
                complement.apply(
                    lambda row, own_keys=own_keys: (
                        (
                            int(row["residue_index"]),
                            int(row["charge_high"]),
                            int(row["charge_low"]),
                        )
                        not in own_keys
                    ),
                    axis=1,
                )
            ]
            merged = pd.concat([timeseries, complement_only], ignore_index=True)
        overlays.append((label, merged))
    return overlays


def _pka_timeseries_plot_ylim(
    series: Sequence[pd.Series], *, margin: float = 0.5
) -> tuple[float, float]:
    """Y-axis limits that include every overlaid replica series."""
    parts = [s for s in series if not s.empty]
    if not parts:
        return 0.0, 14.0
    combined = pd.concat(parts)
    reasonable = combined[(combined >= 0.0) & (combined <= 14.0)]
    if reasonable.empty:
        return 0.0, 14.0
    lo = max(0.0, float(reasonable.min()) - margin)
    hi = min(14.0, float(reasonable.max()) + margin)
    if hi <= lo:
        hi = min(14.0, lo + 1.0)
    return lo, hi


def _pka_timeseries_ylim(pka: pd.Series, *, margin: float = 2.0) -> tuple[float, float]:
    """Return y-axis limits that ignore early diverged pKa fits."""
    reasonable = pka[(pka >= 0.0) & (pka <= 14.0)]
    if reasonable.empty:
        return 0.0, 14.0

    tail = reasonable.iloc[len(reasonable) // 3 :]
    if tail.empty:
        tail = reasonable

    center = float(tail.median())
    iqr = float(tail.quantile(0.75) - tail.quantile(0.25))
    half_range = max(margin, 0.5 * iqr + 0.5)
    lo = max(0.0, center - half_range)
    hi = min(14.0, center + half_range)
    if hi <= lo:
        lo = max(0.0, center - margin)
        hi = min(14.0, center + margin)
    return lo, hi


def plot_pka_timeseries(
    pka_timeseries: pd.DataFrame,
    output_path: str | Path,
    *,
    cph: ConstantPH | None = None,
    filename: str = "pka_timeseries.png",
    overlay_timeseries: Sequence[tuple[str, pd.DataFrame]] | None = None,
) -> Path:
    """Plot cumulative pKa estimates over production time.

    Each charge transition gets its own subplot. When ``overlay_timeseries``
    is provided, each replica curve is drawn underneath the main series and a
    shaded band spans the per-time min/max across replicas.

    Parameters
    ----------
    pka_timeseries : pandas.DataFrame
        Output of :func:`compute_pka_timeseries`.
    output_path : str | Path
        Directory in which to write the figure.
    cph : ConstantPH, optional
        When provided, draw a horizontal line at the model pKa from each
        titratable residue reference for every charge transition.
    filename : str, optional
        Output image filename.
    overlay_timeseries : sequence of (str, pandas.DataFrame), optional
        Additional cumulative pKa series to overlay, e.g. one entry per
        replica. Each DataFrame must use the same schema as
        ``pka_timeseries``.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / filename

    if pka_timeseries.empty:
        return figure_path

    transitions = list(
        pka_timeseries.groupby(
            ["residue_index", "charge_high", "charge_low", "label"],
            sort=False,
        )
    )
    n_transitions = len(transitions)
    ncols = min(3, n_transitions)
    nrows = int(np.ceil(n_transitions / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        squeeze=False,
        sharex=True,
    )

    for ax, ((residue_index, charge_high, charge_low, label), group) in zip(
        axes.flat, transitions, strict=False
    ):
        group_sorted = group.sort_values("time_ns")
        replica_series: list[pd.Series] = []
        overlay_groups: list[tuple[str, pd.DataFrame]] = []
        if overlay_timeseries:
            for replica_label, overlay_df in overlay_timeseries:
                overlay_group = overlay_df[
                    (overlay_df["residue_index"] == residue_index)
                    & (overlay_df["charge_high"] == charge_high)
                    & (overlay_df["charge_low"] == charge_low)
                ].sort_values("time_ns")
                if overlay_group.empty:
                    continue
                overlay_groups.append((replica_label, overlay_group))
                replica_series.append(
                    overlay_group.set_index("time_ns")["pka"].rename(replica_label)
                )

            if len(replica_series) >= 2:
                replica_bounds = pd.concat(replica_series, axis=1)
                ax.fill_between(
                    replica_bounds.index,
                    replica_bounds.min(axis=1),
                    replica_bounds.max(axis=1),
                    alpha=0.12,
                    color="gray",
                    label="replica overlap",
                    zorder=1,
                )

        pooled_label = "average" if overlay_timeseries else None
        ax.plot(
            group_sorted["time_ns"],
            group_sorted["pka"],
            marker="o" if not overlay_timeseries else None,
            markersize=3,
            linewidth=1.5 if overlay_timeseries else 2,
            linestyle=":" if overlay_timeseries else "-",
            color="black",
            label=pooled_label,
            zorder=4,
        )
        if group_sorted["pka_err"].notna().any():
            err = group_sorted["pka_err"].clip(upper=5.0)
            ax.fill_between(
                group_sorted["time_ns"],
                group_sorted["pka"] - err,
                group_sorted["pka"] + err,
                alpha=0.15,
                color="black",
                zorder=2,
            )

        if overlay_groups:
            overlay_colors = ("#2166ac", "#d6604d", "#4daf4a", "#984ea3")
            for replica_i, (replica_label, overlay_group) in enumerate(overlay_groups):
                ax.plot(
                    overlay_group["time_ns"],
                    overlay_group["pka"],
                    linewidth=2.5,
                    linestyle="-",
                    color=overlay_colors[replica_i % len(overlay_colors)],
                    label=replica_label,
                    zorder=5,
                )

        model_pka = None
        if cph is not None:
            titration = cph.titrations.get(residue_index)
            if titration is not None:
                model_pka = titration.reference.macro_pkas_by_charge_transition.get(
                    (charge_high, charge_low)
                )
        if model_pka is not None:
            ax.axhline(
                model_pka,
                color="#e66101",
                linestyle="--",
                linewidth=1.5,
                label=f"model pKa ({model_pka:.2f})",
                zorder=3,
            )

        if overlay_timeseries and overlay_groups:
            ylim_series = [group_sorted["pka"], *replica_series]
            ylo, yhi = _pka_timeseries_plot_ylim(ylim_series)
        else:
            ylo, yhi = _pka_timeseries_ylim(group_sorted["pka"])
        if model_pka is not None:
            ylo = min(ylo, model_pka - 0.5)
            yhi = max(yhi, model_pka + 0.5)
        ax.set_ylim(ylo, yhi)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("pKa estimate")
        ax.grid(True, alpha=0.3)
        if overlay_timeseries or model_pka is not None:
            ax.legend(loc="best", fontsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("Production time (ns)")

    for ax in axes.flat[n_transitions:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def plot_microstate_populations(
    populations: dict[int, pd.DataFrame],
    cph: ConstantPH,
    output_path: str | Path,
) -> list[Path]:
    """Plot microstate population vs pH for each titratable residue.

    Parameters
    ----------
    populations : dict[int, pandas.DataFrame]
        Output of :func:`compute_populations`.
    cph : ConstantPH
        The constant-pH model that produced ``populations``.
    output_path : str | Path
        Directory in which to write the figures.

    Returns
    -------
    list[pathlib.Path]
        Paths to the saved figures.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []

    for residue_index, pop_df in populations.items():
        if pop_df.empty or pop_df.index.nunique() < 2:
            # A single-pH run yields one point on the pH axis; nothing to plot.
            continue

        residue = next(r for r in cph.explicitTopology.residues() if r.index == residue_index)
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = pop_df.sort_index()
        for col in plot_df.columns:
            ax.plot(plot_df.index, plot_df[col], label=col, linewidth=2)

        ax.set_xlabel("pH")
        ax.set_ylabel("Population")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(residue_label(residue))
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        figure_path = output_dir / f"populations_{residue_label_slug(residue)}.png"
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
        figure_paths.append(figure_path)

    return figure_paths


def plot_titration_curves(
    df: pd.DataFrame,
    pkas: dict[int, dict[tuple[int, int], tuple[float, float]]],
    cph: ConstantPH,
    output_path: str | Path,
    *,
    filename: str = "titration_curves.png",
) -> Path | None:
    """Plot fraction-protonated titration curves vs pH, overlaying all residues.

    For each adjacent charge transition of each titratable residue, the per-pH
    fraction of samples in the higher-charge (more protonated) state is drawn as
    markers, computed identically to :func:`calculate_pkas` so the points match
    the fit. The fitted Henderson-Hasselbalch curve is overlaid and the fitted
    macroscopic pKa is shown in the legend, mirroring a classic pH-REMD
    titration plot. All transitions share one axes.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form sampling output with a ``'ph'`` column and one column per
        titratable residue index.
    pkas : dict
        Output of :func:`calculate_pkas` (``{residue_index: {(c_high, c_low):
        (pka, pka_err)}}``). Supplies the fitted pKa for each fit curve.
    cph : ConstantPH
        The constant-pH model that produced ``df``.
    output_path : str | Path
        Directory in which to write the figure.
    filename : str, optional
        Output image filename.

    Returns
    -------
    pathlib.Path | None
        Path to the saved figure, or ``None`` when there is nothing to plot
        (e.g. a single-pH run, or no titratable transitions were sampled).
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / filename

    if df.empty or df["ph"].nunique() < 2:
        # A single-pH run yields one point on the pH axis; no curve to draw.
        return None

    curves: list[tuple[Any, int, int, tuple[float, float] | None, pd.DataFrame]] = []
    for residue_index, titration in cph.titrations.items():
        charges = titration.reference.charges
        residue = next(r for r in cph.explicitTopology.residues() if r.index == residue_index)
        unique_charges = sorted(set(charges), reverse=True)
        charge_series = df[residue_index].map(lambda i, ch=charges: ch[i])
        for c_high, c_low in itertools.pairwise(unique_charges):
            mask = charge_series.isin([c_high, c_low])
            if not mask.any():
                continue
            sub_charge = charge_series[mask]
            if set(sub_charge.unique()) != {c_high, c_low}:
                # e.g. a serial replica that never visits one charge state.
                continue
            frac = (
                df.loc[mask]
                .assign(_charge=sub_charge)
                .groupby("ph")["_charge"]
                .apply(lambda x, ch=c_high: float(np.mean(x == ch)))
                .reset_index(name="f_high")
                .sort_values("ph")
            )
            pka_entry = pkas.get(residue_index, {}).get((c_high, c_low))
            curves.append((residue, c_high, c_low, pka_entry, frac))

    if not curves:
        return None

    ph_lo = float(df["ph"].min())
    ph_hi = float(df["ph"].max())
    ph_grid = np.linspace(ph_lo, ph_hi, 200)
    markers = ("+", "x", "*", "o", "s", "^", "v", "D")
    colors = ("#d6604d", "#4393c3", "#4daf4a", "#984ea3", "#ff7f00", "#a65628")
    # Disambiguate the legend when a residue contributes more than one transition.
    residue_indices = [residue.index for residue, *_ in curves]
    multi_transition = len(set(residue_indices)) < len(curves)

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (residue, c_high, c_low, pka_entry, frac) in enumerate(curves):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        base_label = residue_label(residue)
        if multi_transition:
            base_label = f"{base_label} ({c_high:+d}/{c_low:+d})"
        label = f"{base_label} (pKa={pka_entry[0]:.2f})" if pka_entry is not None else base_label
        ax.plot(
            frac["ph"],
            frac["f_high"],
            linestyle="none",
            marker=marker,
            markersize=8,
            markeredgewidth=1.5,
            color=color,
            label=label,
            zorder=3,
        )
        if pka_entry is not None:
            ax.plot(
                ph_grid,
                henderson_hasselbalch(ph_grid, pka_entry[0]),
                color="black",
                linewidth=1.2,
                zorder=2,
            )

    ax.set_xlabel("pH")
    ax.set_ylabel("Fraction protonated")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(ph_lo, ph_hi)
    ax.set_title("Titration curves")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def analyze_cph_results(
    df: pd.DataFrame,
    cph: ConstantPH,
    output_path: str | Path,
    *,
    sample_interval_ns: float,
    verbose: bool = True,
    overlay_timeseries: Sequence[tuple[str, pd.DataFrame]] | None = None,
    replica_dfs: Sequence[pd.DataFrame] | None = None,
    skip_pka: bool = False,
) -> dict[str, Any]:
    """Compute pKa/population analyses and write CSV/plot outputs."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "results.csv", index=False)

    pkas: dict[int, dict[tuple[int, int], tuple[float, float]]] = {}
    if not skip_pka:
        pkas = calculate_pkas(df, cph)
    populations = compute_populations(df, cph)
    joint_populations = compute_joint_populations(df, cph)
    residue_correlations = compute_residue_correlations(joint_populations, cph)
    if skip_pka:
        pka_timeseries = pd.DataFrame(
            columns=[
                "time_ns",
                "n_samples",
                "residue_index",
                "charge_high",
                "charge_low",
                "label",
                "pka",
                "pka_err",
            ]
        )
    elif replica_dfs is not None:
        pka_timeseries = compute_pka_timeseries_from_replicas(
            replica_dfs, cph, sample_interval_ns=sample_interval_ns
        )
    else:
        pka_timeseries = compute_pka_timeseries(df, cph, sample_interval_ns=sample_interval_ns)

    if not joint_populations.empty:
        state_idx = {name: i for i, name in enumerate(joint_populations.columns)}
        long_joint = (
            joint_populations.rename_axis("ph")
            .reset_index()
            .melt(id_vars="ph", var_name="state_name", value_name="population")
        )
        long_joint.insert(1, "state_idx", long_joint["state_name"].map(state_idx))
        long_joint = long_joint.sort_values(["ph", "state_idx"]).reset_index(drop=True)[
            ["ph", "state_idx", "state_name", "population"]
        ]
        long_joint.to_csv(output_dir / "joint_populations.csv", index=False)
    if not residue_correlations.empty:
        residue_correlations.to_csv(output_dir / "residue_correlations.csv", index=False)

    population_frames: list[pd.DataFrame] = []
    pka_rows: list[dict[str, Any]] = []
    for residue_index, charge_pka_map in pkas.items():
        titration = cph.titrations[residue_index]
        residue = next(r for r in cph.explicitTopology.residues() if r.index == residue_index)
        variant_names = titration.variant_names
        charges = titration.charges
        for (c_high, c_low), (pka, pka_err) in charge_pka_map.items():
            high_names = [n for n, c in zip(variant_names, charges, strict=False) if c == c_high]
            low_names = [n for n, c in zip(variant_names, charges, strict=False) if c == c_low]
            pka_rows.append(
                {
                    "residue_index": residue_index,
                    "residue_name": residue.name,
                    "residue_id": residue.id,
                    "chain_id": (residue.chain.id or "").strip(),
                    "charge_high": c_high,
                    "charge_low": c_low,
                    "high_variants": "/".join(high_names),
                    "low_variants": "/".join(low_names),
                    "pka": pka,
                    "pka_err": pka_err,
                }
            )
            if verbose:
                print(
                    f"  residue {residue_label(residue)} "
                    f"{'/'.join(high_names)} ({c_high:+d}) -> "
                    f"{'/'.join(low_names)} ({c_low:+d}): "
                    f"pKa = {pka:.3f} +/- {pka_err:.3f}"
                )
        if verbose:
            print(f"populations for residue {residue_label(residue)}:")
            print(populations[residue_index])

    for residue_index, populations_df in populations.items():
        residue = next(r for r in cph.explicitTopology.residues() if r.index == residue_index)
        state_idx = {name: i for i, name in enumerate(populations_df.columns)}
        frame = (
            populations_df.rename_axis("ph")
            .reset_index()
            .melt(id_vars="ph", var_name="state_name", value_name="population")
        )
        frame.insert(1, "residue_index", residue_index)
        frame.insert(2, "residue_name", residue.name)
        frame.insert(3, "residue_id", residue.id)
        frame.insert(4, "chain_id", (residue.chain.id or "").strip())
        frame.insert(5, "state_idx", frame["state_name"].map(state_idx))
        population_frames.append(frame)

    if population_frames:
        long_pops = (
            pd.concat(population_frames, ignore_index=True)
            .sort_values(["ph", "residue_index", "state_idx"])
            .reset_index(drop=True)[
                [
                    "ph",
                    "residue_index",
                    "residue_name",
                    "residue_id",
                    "chain_id",
                    "state_idx",
                    "state_name",
                    "population",
                ]
            ]
        )
        long_pops.to_csv(output_dir / "populations.csv", index=False)
        plot_microstate_populations(populations, cph, output_dir)
    if pka_rows:
        pd.DataFrame(pka_rows).to_csv(output_dir / "pkas.csv", index=False)
    if not skip_pka:
        plot_titration_curves(df, pkas, cph, output_dir)
    if not pka_timeseries.empty:
        pka_timeseries.to_csv(output_dir / "pka_timeseries.csv", index=False)
        plot_pka_timeseries(
            pka_timeseries,
            output_dir,
            cph=cph,
            overlay_timeseries=overlay_timeseries,
        )

    if verbose:
        print("\nJoint microstate populations:")
        print(joint_populations.T)
        print("\nResidue charge correlations:")
        print(residue_correlations)

    return {
        "results": df,
        "pkas": pkas,
        "populations": populations,
        "joint_populations": joint_populations,
        "residue_correlations": residue_correlations,
        "pka_timeseries": pka_timeseries,
    }
