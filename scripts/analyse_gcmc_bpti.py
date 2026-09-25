#!/usr/bin/env python
"""Compare run directories of benchmark_gcmc_bpti.py.

Per run: mean N after burn-in, statistical inefficiency g, effective sample size,
standard error, P(N), accepted moves per trial and wall time per effective sample.
Runs of one sampler, batch size and start are pooled; each pair of pools with the
same mode and start is compared as a difference of means with a z-score and a
chi-square test on P(N) with effective counts. The pooled SE is the larger of the
within-run SE and the SE of the per-run means across seeds. A run with fewer than
MIN_CHANGES changes of N has not mixed: it is flagged and left out of the pools.
In md mode, if grand is importable, grand's own cluster analysis runs on each
trajectory. Writes summary.md and pn.png to --out.

grand's insertion orientation is not uniform over SO(3), so a grand-vs-parallel
difference is not by itself evidence against the batched sampler. Compare
grand-haar with parallel, or parallel at batch size 1 with a large batch size.

    python scripts/analyse_gcmc_bpti.py runs/* --out runs/summary
"""

import itertools
import json
import math
from pathlib import Path

import click
import numpy as np
import pandas as pd
from pymbar import timeseries
from scipy import stats

try:
    import grand
except ImportError:
    grand = None
try:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

MIN_CHANGES = 20
CAVEAT = (
    "grand's insertion orientation is not uniform over SO(3). A difference between grand "
    "(without -haar) and parallel is not by itself evidence against the batched sampler."
)


def _run(path, burn_in, series):
    settings = json.loads((path / "settings.json").read_text())
    if series == "cycle":
        ns = pd.read_csv(path / "cycles.csv")["N"].to_numpy(float)
    else:
        ns = np.load(path / "Ns.npy").astype(float)
    ns = ns[int(burn_in * len(ns)) :]
    try:
        g = timeseries.statisticalInefficiency(ns)
        se = ns.std() * math.sqrt(g / len(ns))
    except timeseries.ParameterError:  # a constant series: the error is unknown
        g, se = float(len(ns)), math.nan
    n_eff = len(ns) / g
    p = np.bincount(ns.astype(int)) / len(ns)
    haar = "-haar" * settings.get("grand_haar", False)
    group = f"{settings['sampler']}{haar}-b{settings['batch_size']}"
    key = (settings["mode"], settings["empty_sphere"])
    t = path / "timing.json"
    timing = json.loads(t.read_text()) if t.exists() else {}
    return {
        "label": f"{group}-{settings['mode']}-s{settings['seed']}" + "-empty" * key[1],
        "group": group, "key": key, "n": len(ns), "changes": int(np.count_nonzero(np.diff(ns))),
        "mean": ns.mean(), "g": g, "n_eff": n_eff, "se": se, "p": p,
        "ms_per_trial": timing.get("gcmc_ms_per_trial", math.nan),
        "acc_per_trial": timing.get("n_accepted", math.nan) / timing.get("n_moves", math.nan),
        "s_per_eff": (timing.get("gcmc_s", math.nan) + (timing.get("md_s") or 0)) / n_eff,
        "path": path, "settings": settings,
    }  # fmt: skip


def _clusters(run):
    """Cluster occupancies from grand's pipeline, or None if it cannot run."""
    d, s = run["path"], run["settings"]
    if grand is None or run["key"][0] != "md":
        return None
    top = str(d / "topology.pdb")
    trj = grand.utils.shift_ghost_waters(
        str(d / "ghosts.txt"), topology=top, trajectory=str(d / "traj.dcd")
    )
    trj = grand.utils.recentre_traj(t=trj, resname="TYR", resid=10)
    grand.utils.align_traj(t=trj, output=str(d / "aligned.dcd"))
    try:
        grand.utils.cluster_waters(
            top, str(d / "aligned.dcd"), s["sphere_radius_a"], ref_atoms=s["ref_atoms"],
            cutoff=2.4, output=str(d / "clusters.pdb"),
        )  # fmt: skip
    except ValueError:  # fewer than two water observations in the sphere
        return None
    lines = (d / "clusters.pdb").read_text().splitlines()
    return [float(line[54:60]) for line in lines if line.startswith("ATOM")]


def _pn(p):
    return " ".join(f"{n}:{x:.3f}" for n, x in enumerate(p) if x > 0)


@click.command()
@click.argument("runs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--burn-in", default=0.2, help="Fraction of each series to discard.")
@click.option("--series", type=click.Choice(["trial", "cycle"]), default="trial")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(runs, burn_in, series, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = [_run(r, burn_in, series) for r in runs]
    md = [f"# GCMC comparison\n\n{CAVEAT}\n\nBurn-in {burn_in:.0%}, N per {series}.\n"]
    md.append(
        "| run | samples | changes | mean N | g | n_eff | SE | ms/trial | acc/trial "
        "| s per eff. sample | P(N) |\n" + "|---" * 11 + "|"
    )
    for r in rows:
        flag = " (not mixed, left out)" * (r["changes"] < MIN_CHANGES)
        md.append(
            f"| {r['label']}{flag} | {r['n']} | {r['changes']} | {r['mean']:.3f} | {r['g']:.1f} "
            f"| {r['n_eff']:.0f} | {r['se']:.3f} | {r['ms_per_trial']:.3f} "
            f"| {r['acc_per_trial']:.4f} | {r['s_per_eff']:.3g} | {_pn(r['p'])} |"
        )
    mixed = [r for r in rows if r["changes"] >= MIN_CHANGES]
    width = max(len(r["p"]) for r in rows)
    for key in sorted({r["key"] for r in mixed}):
        # Pool the runs of one group (sampler and batch size), weighted by n_eff.
        pools = {}
        for group in sorted({r["group"] for r in mixed if r["key"] == key}):
            pool = [r for r in mixed if r["key"] == key and r["group"] == group]
            w = np.array([r["n_eff"] for r in pool])
            means, ses = np.array([r["mean"] for r in pool]), np.array([r["se"] for r in pool])
            within = math.sqrt(((w * ses) ** 2).sum()) / w.sum()
            between = means.std(ddof=1) / math.sqrt(len(pool)) if len(pool) > 1 else 0.0
            counts = np.zeros(width)
            for r in pool:
                counts[: len(r["p"])] += r["p"] * r["n_eff"]
            pools[group] = ((w * means).sum() / w.sum(), np.max([within, between]), counts)
        start = "empty sphere" if key[1] else "full sphere"
        for a, b in itertools.combinations(pools, 2):
            (ma, sa, ca), (mb, sb, cb) = pools[a], pools[b]
            se = math.hypot(sa, sb)
            z = (ma - mb) / se if se > 0 else math.nan  # nan: an SE is unknown
            table = np.array([ca, cb])[:, (ca + cb) > 0]
            chi = "not done (one bin)"
            if table.shape[1] > 1:
                chi2, pval, dof, expected = stats.chi2_contingency(table)
                chi = (
                    f"chi2 = {chi2:.2f}, dof = {dof}, p = {pval:.3g}"
                    if expected.min() >= 5
                    else "not done (an expected effective count is below 5)"
                )
            md.append(
                f"\n## {key[0]}, {start}: {a} vs {b}\n\n- mean N: {a} {ma:.3f} +- {sa:.3f}, "
                f"{b} {mb:.3f} +- {sb:.3f}\n- difference {ma - mb:+.3f} +- {se:.3f}, "
                f"z = {z:+.2f}\n- P(N), effective counts: {chi}"
            )
    for r in rows:
        occ = _clusters(r)
        if occ is not None:
            top = ", ".join(f"{x:.2f}" for x in occ[:8])
            md.append(f"\n{r['label']}: {len(occ)} clusters, occupancies {top}")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    click.echo("\n".join(md))
    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in rows:
            ax.plot(np.arange(len(r["p"])), r["p"], marker="o", label=r["label"])
        ax.set(xlabel="N", ylabel="P(N)")
        ax.legend(fontsize=7)
        fig.savefig(out / "pn.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
