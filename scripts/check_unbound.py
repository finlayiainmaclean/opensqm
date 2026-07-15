#!/usr/bin/env python3
"""Explicit unbound-state check vs the Einstein-Smoluchowski estimate.

Runs a single long ligand-in-water simulation at 300 K, measures the mean COM
escape time to the absorbing boundary, and converts it to an unbound free
energy with the SI Eq. SI-4 convention (``N_replicas * t / dt_bound``). Compares
against the Einstein-Smoluchowski estimate and the paper's reported range for
the Wang dataset (-2.96 .. -3.46, mean -3.16 kcal/mol).

Example:
    pixi run python scripts/check_unbound.py data/inputs/Wang/BACE/13b.sdf
"""

import argparse
import math
import time

from openmm import unit
from rdkit import Chem

from opensqm.md.equilibrate import EquilibrationSettings
from opensqm.modbind.config import ModBindDGSettings
from opensqm.modbind.escape import run_unbound_escape
from opensqm.modbind.reweight import einstein_smoluchowski_unbound, rt_kcal
from opensqm.modbind.states import build_unbound_state

# SI Eq. SI-4 reference parameters.
N_REPLICAS = 32
DT_BOUND_NS = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ligand", help="Ligand SDF/MOL file.")
    parser.add_argument(
        "--sim-time-ns", type=float, default=5.0, help="Unbound simulation length (ns)."
    )
    parser.add_argument("--temperature", type=float, default=300.0, help="Unbound temperature (K).")
    parser.add_argument(
        "--boundary", type=float, default=5.0, help="Absorbing boundary radius (A)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lig = Chem.MolFromMolFile(args.ligand, removeHs=True)
    if lig is None:
        raise SystemExit(f"Could not read ligand from {args.ligand}")
    print("ligand SMILES:", Chem.MolToSmiles(lig), flush=True)
    lig = Chem.AddHs(lig, addCoords=True)

    equilibration = EquilibrationSettings(
        warmup_time=20 * unit.picoseconds,
        npt_time=80 * unit.picoseconds,
    )
    config = ModBindDGSettings(
        unbound_temperature=args.temperature * unit.kelvin,
        unbound_max_time=args.sim_time_ns * unit.nanosecond,
        unbound_target_escapes=10**9,  # don't stop early; run the full sim
        unbound_frame_interval=1 * unit.picosecond,
        bound_frame_interval=DT_BOUND_NS * 1000 * unit.picoseconds,
        integrator_step_size=0.002 * unit.picosecond,
        absorbing_boundary_radius=args.boundary,
        equilibration_config=equilibration,
    )

    t0 = time.time()
    print("building + equilibrating unbound state ...", flush=True)
    state = build_unbound_state(lig, equilibration_config=config.equilibration_config)
    print(
        f"  done in {time.time() - t0:.0f}s; running {args.sim_time_ns} ns unbound sim",
        flush=True,
    )

    segments, times_steps, _converged = run_unbound_escape(state, config)

    rt = rt_kcal(300.0)
    step_ps = config.integrator_step_size.value_in_unit(unit.picosecond)
    n_esc = len(segments)
    t_mean_ns = float(times_steps.mean()) * step_ps / 1000.0 if n_esc else float("nan")

    g_explicit = -rt * math.log(N_REPLICAS * t_mean_ns / DT_BOUND_NS) if n_esc else float("nan")
    _, g_einstein = einstein_smoluchowski_unbound(ModBindDGSettings(n_replicas=N_REPLICAS), rt=rt)

    print("==== RESULT ====", flush=True)
    print(f"escapes observed         : {n_esc}")
    print(f"mean escape time         : {t_mean_ns:.4f} ns (Einstein assumes 0.0833 ns)")
    print(f"G_unbound (explicit t)   : {g_explicit:.3f} kcal/mol")
    print(f"G_unbound (Einstein)     : {g_einstein:.3f} kcal/mol")
    print("paper explicit range     : -2.96 .. -3.46 (mean -3.16) kcal/mol")
    print(f"total wall time          : {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
