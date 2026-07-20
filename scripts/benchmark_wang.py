#!/usr/bin/env python
"""Benchmark ModBinddG (and the MMGBSA protomer-funnel score) against a Wang
FEP-set target.

Given a system name (e.g. ``Tyk2``) this:

* reads the ligands + experimental affinities from
  ``data/inputs/Wang/<System>/protein.csv`` and the fixed protein referenced in
  that CSV,
* for each ligand runs a full ModBinddG calculation (:func:`run_modbind`) at the
  requested temperature / replica count -- which also yields the MMGBSA
  protomer-funnel score (interaction energy + protonation penalty) as its
  equilibration by-product,
* converts experimental ``pX`` to ΔG (``ΔG = -RT ln(10) pX``),
* and reports, updated after EVERY ligand so you get an early signal:
    - ModBinddG:  Pearson R, R^2, absolute MUE and RMSE (kcal/mol) vs experiment
    - MMGBSA:     Pearson R, R^2 vs experiment (a score, so no absolute MUE)
    - per-ligand diagnostics: c_min / c_boundary / ΔG_well / n_escapes so you can
      see whether the bound well was actually sampled.

Everything is cached under ``--cache-dir`` (one subdir per ligand) via
``run_modbind``'s resume, and the running table is written to
``benchmark_<System>.csv`` there, so the script is fully resumable and can run
outside an interactive session:

    pixi run python scripts/benchmark_wang.py Tyk2 --platform mps

Paper reference (Sinko et al., SI Fig S7, raw ModBinddG): Tyk2 R^2 = 0.57,
RMSE = 1.82, MUE = 1.58 kcal/mol -- albeit with many more replicas than 4.

Use ``--mmgbsa-only`` for a fast (~minutes/ligand) MMGBSA-only correlation pass
before committing to the long ModBinddG run.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from openmm import unit

from opensqm.md.platforms import set_platform
from opensqm.modbind.config import ModBindDGSettings

REPO_ROOT = Path(__file__).resolve().parent.parent
WANG_ROOT = REPO_ROOT / "data" / "inputs" / "Wang"


def exp_dg_from_px(value: float) -> float:
    """Experimental ΔG (kcal/mol) from the CSV 'pX' column.

    NB: despite its name, the 'pX' column stores -ΔG in kcal/mol (e.g. 9.78 means
    ΔG = -9.78 kcal/mol), NOT a pKi/pIC50 -- so ΔG is simply its negation.
    """
    return -float(value)


def metrics(pred: np.ndarray, exp: np.ndarray) -> dict:
    """Pearson R, R^2, MUE and RMSE of ``pred`` vs ``exp`` over finite pairs."""
    pred = np.asarray(pred, dtype=np.float64)
    exp = np.asarray(exp, dtype=np.float64)
    ok = np.isfinite(pred) & np.isfinite(exp)
    pred, exp = pred[ok], exp[ok]
    n = int(pred.size)
    out = {"n": n, "r": math.nan, "r2": math.nan, "mue": math.nan, "rmse": math.nan}
    if n >= 1:
        out["mue"] = float(np.mean(np.abs(pred - exp)))
        out["rmse"] = float(np.sqrt(np.mean((pred - exp) ** 2)))
    if n >= 2 and np.std(pred) > 0 and np.std(exp) > 0:
        r = float(np.corrcoef(pred, exp)[0, 1])
        out["r"], out["r2"] = r, r * r
    return out


def load_system(system: str) -> tuple[Path, pd.DataFrame]:
    """Return (fixed protein path, ligand dataframe with columns lig_id/sdf/pX/exp_dg)."""
    sysdir = WANG_ROOT / system
    csv = sysdir / "protein.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No protein.csv for system '{system}' at {csv}")
    df = pd.read_csv(csv)
    protein = REPO_ROOT / df.iloc[0]["protein"]
    rows = pd.DataFrame(
        {
            "lig_id": [Path(p).stem for p in df["ligand"]],
            "sdf": [str(REPO_ROOT / p) for p in df["ligand"]],
            "pX": df["pX"].astype(float),
            "exp_dg": [exp_dg_from_px(x) for x in df["pX"]],
        }
    )
    return protein, rows


def _log_running(rows: list[dict]) -> None:
    """Print running ModBinddG + MMGBSA metrics after the ligands seen so far."""
    rdf = pd.DataFrame(rows)
    exp = rdf["exp_dg"].to_numpy()
    mb = metrics(rdf["modbind_dg"].to_numpy(), exp)
    mg = metrics(rdf["mmgbsa_score"].to_numpy(), exp)
    logger.info(
        f"  RUNNING ({len(rows)} ligands)  "
        f"ModBind: R2={mb['r2']:.2f} R={mb['r']:.2f} MUE={mb['mue']:.2f} RMSE={mb['rmse']:.2f} | "
        f"MMGBSA: R2={mg['r2']:.2f} R={mg['r']:.2f}   (paper Tyk2: R2=0.57, MUE=1.58)"
    )


def run_modbind_pass(
    system: str, protein: Path, ligands: pd.DataFrame, config, cache: Path
) -> None:
    """Full ModBinddG + MMGBSA benchmark, one ligand at a time, resumable."""
    from opensqm.modbind.run_modbind import run_modbind

    results_csv = cache / f"benchmark_{system}.csv"
    # Keyed by ligand so re-running at a higher --n-replicas UPDATES the row
    # rather than appending a duplicate.
    rows_by_id: dict[str, dict] = {}
    if results_csv.exists():
        prev = pd.read_csv(results_csv)
        rows_by_id = {r["lig_id"]: r for r in prev.to_dict("records")}
        # Refresh exp_dg from the current conversion, in case it was corrected
        # between runs (so cached rows don't keep a stale experimental value).
        exp_by_id = dict(zip(ligands["lig_id"], ligands["exp_dg"], strict=False))
        for lid, rec in rows_by_id.items():
            if lid in exp_by_id:
                rec["exp_dg"] = exp_by_id[lid]
        logger.info(f"Resuming: {len(rows_by_id)} ligand(s) already in {results_csv}")

    for _, lig in ligands.iterrows():
        lig_id = lig["lig_id"]
        cached = rows_by_id.get(lig_id)
        # Skip only if the cached result already used >= the requested replicas.
        # Otherwise re-run: run_modbind reuses the cached replicas (and the
        # equilibration + unbound trajectory) and only runs the additional ones.
        if cached is not None and int(cached.get("n_bound_escapes") or 0) >= config.n_replicas:
            logger.info(
                f"[{lig_id}] cached with {int(cached['n_bound_escapes'])} replicas "
                f">= {config.n_replicas} requested, skipping"
            )
            continue
        n_have = int(cached["n_bound_escapes"]) if cached is not None else 0
        extra = f" (have {n_have}, extending to {config.n_replicas})" if n_have else ""
        logger.info(
            f"=== {lig_id}  (exp ΔG = {lig['exp_dg']:.2f} kcal/mol, pX={lig['pX']:.2f}){extra} ==="
        )
        try:
            res = run_modbind(str(protein), lig["sdf"], str(cache / lig_id), config=config)
        except Exception as exc:  # one bad ligand must not kill the benchmark
            logger.exception(f"[{lig_id}] run_modbind FAILED: {exc}")
            continue

        rec = {
            "lig_id": lig_id,
            "pX": lig["pX"],
            "exp_dg": lig["exp_dg"],
            "modbind_dg": res.get("delta_g"),
            "ci_low": res.get("ci_low"),
            "ci_high": res.get("ci_high"),
            "mmgbsa_score": res.get("mmgbsa_score"),
            "c_min": res.get("c_min"),
            "c_boundary": res.get("c_boundary"),
            "c_min_radius": res.get("c_min_radius"),
            "delta_g_well": res.get("delta_g_well"),
            "n_bound_escapes": res.get("n_bound_escapes"),
            "bound_population": res.get("bound_population"),
            "unbound_population": res.get("unbound_population"),
        }
        rows_by_id[lig_id] = rec
        rows = list(rows_by_id.values())
        pd.DataFrame(rows).to_csv(results_csv, index=False)

        undersampled = " [LOW c_min -- well under-sampled?]" if (rec["c_min"] or 0) < 500 else ""
        logger.info(
            f"[{lig_id}] ModBind ΔG={rec['modbind_dg']:.2f} "
            f"[{rec['ci_low']:.2f},{rec['ci_high']:.2f}]  MMGBSA={rec['mmgbsa_score']:.2f}  |  "
            f"c_min={rec['c_min']} c_boundary={rec['c_boundary']} "
            f"ΔG_well={rec['delta_g_well']:.2f} n_esc={rec['n_bound_escapes']}{undersampled}"
        )
        _log_running(rows)

    _final_report(system, list(rows_by_id.values()), results_csv)


def _report_mmgbsa(rows: list[dict], n_replicas: int, production_ps: float, *, final: bool) -> None:
    """Log running/final correlation of BOTH interaction_energy and mmgbsa_score vs exp."""
    rdf = pd.DataFrame(rows)
    exp = rdf["exp_dg"].to_numpy()
    ie = metrics(rdf["interaction_energy"].to_numpy(), exp)
    mg = metrics(rdf["mmgbsa_score"].to_numpy(), exp)
    if final:
        logger.info("=" * 78)
        logger.info(
            f"FINAL MMGBSA BENCHMARK ({len(rdf)} ligands, {n_replicas}x{production_ps:.0f} ps)"
        )
        logger.info(
            rdf[["lig_id", "exp_dg", "interaction_energy", "protonation_penalty", "mmgbsa_score"]]
            .round(2)
            .to_string(index=False)
        )
        logger.info(f"interaction_energy vs exp ΔG: R2={ie['r2']:.3f}  R={ie['r']:.3f}")
        logger.info(f"mmgbsa_score       vs exp ΔG: R2={mg['r2']:.3f}  R={mg['r']:.3f}")
        logger.info("=" * 78)
    else:
        logger.info(
            f"  RUNNING ({len(rows)} ligands)  "
            f"interaction_energy: R2={ie['r2']:.2f} R={ie['r']:.2f} | "
            f"mmgbsa_score: R2={mg['r2']:.2f} R={mg['r']:.2f}"
        )


def run_mmgbsa_pass(
    system: str,
    protein: Path,
    ligands: pd.DataFrame,
    cache: Path,
    *,
    n_replicas: int,
    production_ps: float,
) -> None:
    """MMGBSA-only benchmark: ``n_replicas`` x ``production_ps`` production per
    ligand (no escape sims), correlating BOTH the raw interaction energy and the
    corrected mmgbsa_score (interaction + protonation penalty) against exp ΔG."""
    from opensqm.md.run_mmgbsa import MMGBSASettings, run_mmgbsa

    results_csv = cache / f"benchmark_{system}_mmgbsa.csv"
    rows_by_id: dict[str, dict] = {}
    if results_csv.exists():
        prev = pd.read_csv(results_csv)
        rows_by_id = {r["lig_id"]: r for r in prev.to_dict("records")}
        logger.info(f"Resuming: {len(rows_by_id)} ligand(s) already in {results_csv}")

    for _, lig in ligands.iterrows():
        lig_id = lig["lig_id"]
        cached = rows_by_id.get(lig_id)
        if cached is not None and pd.notna(cached.get("interaction_energy")):
            logger.info(f"[{lig_id}] cached MMGBSA, skipping")
            continue
        logger.info(
            f"=== {lig_id}  MMGBSA {n_replicas}x{production_ps:.0f}ps  "
            f"(exp ΔG = {lig['exp_dg']:.2f}) ==="
        )
        try:
            result = run_mmgbsa(
                str(protein),
                lig["sdf"],
                output=str(cache / lig_id / "mmgbsa_only"),
                config=MMGBSASettings(
                    production_time=production_ps * unit.picosecond,
                    n_replicas=n_replicas,
                    protomer_ph=7.0,
                    protonation_penalty=3.0 * unit.kilocalories_per_mole,
                ),
            )
            s = result.scores
            rec = {
                "lig_id": lig_id,
                "pX": lig["pX"],
                "exp_dg": lig["exp_dg"],
                "interaction_energy": float(s["interaction_energy"]),
                "interaction_energy_std": float(s["interaction_energy_std"]),
                "protonation_penalty": float(s["protonation_penalty"]),
                "mmgbsa_score": float(s["mmgbsa_score"]),
                "n_replicas": int(s["n_replicas"]),
            }
        except Exception as exc:  # one bad ligand must not kill the benchmark
            logger.exception(f"[{lig_id}] MMGBSA FAILED: {exc}")
            continue
        rows_by_id[lig_id] = rec
        rows = list(rows_by_id.values())
        pd.DataFrame(rows).to_csv(results_csv, index=False)
        logger.info(
            f"[{lig_id}] interaction={rec['interaction_energy']:.2f}"
            f"+/-{rec['interaction_energy_std']:.2f}  penalty={rec['protonation_penalty']:.2f}  "
            f"mmgbsa_score={rec['mmgbsa_score']:.2f}"
        )
        _report_mmgbsa(rows, n_replicas, production_ps, final=False)

    rows = list(rows_by_id.values())
    if rows:
        _report_mmgbsa(rows, n_replicas, production_ps, final=True)
        logger.info(f"Results: {results_csv}")
    else:
        logger.warning("No ligands completed; nothing to report.")


def _final_report(system: str, rows: list[dict], results_csv: Path) -> None:
    if not rows:
        logger.warning("No ligands completed; nothing to report.")
        return
    rdf = pd.DataFrame(rows)
    exp = rdf["exp_dg"].to_numpy()
    mb = metrics(rdf["modbind_dg"].to_numpy(), exp)
    mg = metrics(rdf["mmgbsa_score"].to_numpy(), exp)
    logger.info("=" * 78)
    logger.info(f"FINAL BENCHMARK: {system}  ({len(rdf)} ligands)")
    logger.info(
        rdf[
            [
                "lig_id",
                "exp_dg",
                "modbind_dg",
                "mmgbsa_score",
                "c_min",
                "c_boundary",
                "delta_g_well",
            ]
        ].to_string(index=False)
    )
    logger.info(
        f"ModBinddG : R2={mb['r2']:.3f}  R={mb['r']:.3f} "
        f"MUE={mb['mue']:.2f}  RMSE={mb['rmse']:.2f} kcal/mol"
    )
    logger.info(f"MMGBSA    : R2={mg['r2']:.3f}  R={mg['r']:.3f}  (score; no absolute MUE)")
    logger.info("Paper (Tyk2, raw): R2=0.57  MUE=1.58  RMSE=1.82 kcal/mol")
    logger.info(f"Results: {results_csv}")
    logger.info("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("system", help="Wang target name, e.g. Tyk2 (dir under data/inputs/Wang/).")
    ap.add_argument("--temperature", type=float, default=600.0, help="Bound-state temperature (K).")
    ap.add_argument("--n-replicas", type=int, default=4, help="Bound escape replicas per ligand.")
    ap.add_argument(
        "--unbound-escapes",
        type=int,
        default=32,
        help="Unbound (300 K) escape segments per ligand.",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="Persistent cache/work dir (default: benchmark_runs/<sys>_T<t>_r<n>).",
    )
    ap.add_argument(
        "--platform", choices=["cuda", "mps"], default=None, help="Force OpenMM platform."
    )
    ap.add_argument(
        "--mmgbsa-only", action="store_true", help="MMGBSA-only correlation pass (no escape sims)."
    )
    ap.add_argument(
        "--mmgbsa-replicas", type=int, default=3, help="MMGBSA production replicas (--mmgbsa-only)."
    )
    ap.add_argument(
        "--mmgbsa-production-ps",
        type=float,
        default=100.0,
        help="MMGBSA production time, ps (--mmgbsa-only).",
    )
    args = ap.parse_args()

    set_platform(args.platform)
    protein, ligands = load_system(args.system)
    logger.info(
        f"{args.system}: {len(ligands)} ligands, protein {protein.name}; "
        f"exp ΔG range [{ligands['exp_dg'].min():.2f}, {ligands['exp_dg'].max():.2f}] kcal/mol "
        f"(ΔG = -1 * 'pX' column)"
    )

    # Cache is keyed by system + temperature ONLY (not replica count): the MD
    # replicas are reusable across --n-replicas settings at the same temperature,
    # so `--n-replicas 1` then `--n-replicas 2` reuses replica 0 and runs only 1
    # more. (Different temperatures must use different dirs -- they do, via T.)
    cache = Path(
        args.cache_dir or REPO_ROOT / "benchmark_runs" / f"{args.system}_T{int(args.temperature)}"
    )
    cache.mkdir(parents=True, exist_ok=True)

    config = ModBindDGSettings(
        bound_temperature=args.temperature * unit.kelvin,
        n_replicas=args.n_replicas,
        unbound_target_escapes=args.unbound_escapes,
        bound_box_shape="dodecahedron",  # type: ignore[arg-type]
    )

    if args.mmgbsa_only:
        run_mmgbsa_pass(
            args.system,
            protein,
            ligands,
            cache,
            n_replicas=args.mmgbsa_replicas,
            production_ps=args.mmgbsa_production_ps,
        )
    else:
        run_modbind_pass(args.system, protein, ligands, config, cache)


if __name__ == "__main__":
    main()
