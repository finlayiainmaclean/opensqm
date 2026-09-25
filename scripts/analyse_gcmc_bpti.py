#!/usr/bin/env python
"""Compare run directories of benchmark_gcmc_bpti.py.

Per run: mean N after burn-in, statistical inefficiency g, effective sample size,
standard error and P(N). Per mode: grand against parallel, as a difference of
pooled means with a z-score and a chi-square test on P(N) with effective counts.
In md mode, if grand is importable, grand's own cluster analysis runs on each
trajectory. Writes summary.md and pn.png to --out.

    python scripts/analyse_gcmc_bpti.py runs/* --out runs/summary
"""

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


def _run(path, burn_in, series):
    settings = json.loads((path / "settings.json").read_text())
    ns = np.load(path / "Ns.npy").astype(float)
    if series == "cycle":
        ns = pd.read_csv(path / "cycles.csv")["N"].to_numpy(float)
    ns = ns[int(burn_in * len(ns)) :]
    try:
        g = timeseries.statisticalInefficiency(ns)
    except timeseries.ParameterError:  # a constant series
        g = float(len(ns))
    n_eff = len(ns) / g
    p = np.bincount(ns.astype(int)) / len(ns)
    label = (
        f"{settings['sampler']}-{settings['mode']}-b{settings['batch_size']}-s{settings['seed']}"
    )
    key = (settings["mode"], settings["empty_sphere"])
    timing = json.loads((path / "timing.json").read_text())
    return {
        "label": label + "-empty" * key[1], "sampler": settings["sampler"], "key": key,
        "n": len(ns),
        "mean": ns.mean(), "g": g, "n_eff": n_eff, "se": ns.std() / math.sqrt(n_eff), "p": p,
        "ms_per_trial": timing["gcmc_ms_per_trial"], "path": path, "settings": settings,
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
    md = [f"# GCMC comparison\n\nBurn-in {burn_in:.0%}, N per {series}.\n"]
    md.append("| run | samples | mean N | g | n_eff | SE | ms/trial | P(N) |\n" + "|---" * 8 + "|")
    for r in rows:
        md.append(
            f"| {r['label']} | {r['n']} | {r['mean']:.3f} | {r['g']:.1f} | {r['n_eff']:.0f} "
            f"| {r['se']:.3f} | {r['ms_per_trial']:.3f} | {_pn(r['p'])} |"
        )
    # Pool runs of one sampler, mode and start (full or empty sphere), weighted by n_eff.
    for key in sorted({r["key"] for r in rows}):
        pools = {}
        for name in ("grand", "parallel"):
            pool = [r for r in rows if r["key"] == key and r["sampler"] == name]
            if pool:
                w = np.array([r["n_eff"] for r in pool])
                mean = sum(wi * r["mean"] for wi, r in zip(w, pool, strict=True)) / w.sum()
                se = (
                    math.sqrt(sum((wi * r["se"]) ** 2 for wi, r in zip(w, pool, strict=True)))
                    / w.sum()
                )
                counts = np.zeros(max(len(r["p"]) for r in rows))
                for r in pool:
                    counts[: len(r["p"])] += r["p"] * r["n_eff"]
                pools[name] = (mean, se, counts)
        if len(pools) < 2:
            continue
        (mg, sg, cg), (mp, sp, cp) = pools["grand"], pools["parallel"]
        se = math.hypot(sg, sp)
        z = (mg - mp) / se if se > 0 else math.nan  # nan: both series constant
        table = np.array([cg, cp])[:, (cg + cp) > 0]
        chi2, pval, dof, _ = stats.chi2_contingency(table) if table.shape[1] > 1 else (0, 1, 0, 0)
        start = "empty sphere" if key[1] else "full sphere"
        md.append(
            f"\n## {key[0]}, {start}: grand vs parallel\n\n- mean N: grand {mg:.3f} +- {sg:.3f}, "
            f"parallel {mp:.3f} +- {sp:.3f}\n- difference {mg - mp:+.3f} +- {se:.3f}, "
            f"z = {z:+.2f}\n- P(N), effective counts: chi2 = {chi2:.2f}, dof = {dof}, "
            f"p = {pval:.3g}"
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
