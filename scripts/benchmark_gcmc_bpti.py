#!/usr/bin/env python
"""Run grand's StandardGCMCSphereSampler or the batched GCMCSampler on grand's BPTI system.

Both samplers get the same System, built once with grand's BPTI protocol:
amber14-all + amber14/tip3p, PME, 12 A cutoff, 10 A switch, HBonds, 15 ghost
waters, a 4.2 A sphere on the CA atoms of TYR10 and ASN43, 298 K. ``--mode md``
does ``--md-steps`` of Langevin MD before each GCMC cycle; ``--mode frozen`` does
only GCMC. The run directory gets settings.json, Ns.npy (N after every trial),
cycles.csv, timing.json and, in md mode, traj.dcd and ghosts.txt (grand's format).
Ns.npy and timing.json are rewritten after every cycle, so a killed run keeps its data.

grand's insertion orientation is not uniform over SO(3) (grand.utils.random_rotation_matrix),
so a grand-vs-parallel difference is not by itself evidence against the batched sampler.
For a like-for-like check, run grand with ``--grand-haar`` (uniform rotations), or
compare parallel at ``--batch-size 1`` with parallel at a large batch size.

    python scripts/benchmark_gcmc_bpti.py --sampler parallel --mode frozen --out runs/p0
"""

import json
import time
import urllib.request
from pathlib import Path

import click
import numpy as np
import openmm
from openmm import app, unit
from scipy.spatial.transform import Rotation

from opensqm.gcmc import GCMCSampler, GCMCSettings, add_ghost_waters

try:
    import grand
except ImportError:
    grand = None

PDB_URL = (
    "https://raw.githubusercontent.com/essex-lab/grand/9659ed2/examples/bpti/prod/bpti-equil.pdb"
)
REF_ATOMS = [
    {"name": "CA", "resname": "TYR", "resid": "10"},
    {"name": "CA", "resname": "ASN", "resid": "43"},
]
RADIUS_A, TEMPERATURE = 4.2, 298.0 * unit.kelvin


@click.command()
@click.option("--sampler", "which", type=click.Choice(["grand", "parallel"]), required=True)
@click.option("--mode", type=click.Choice(["frozen", "md"]), default="frozen")
@click.option("--pdb", type=click.Path(path_type=Path), default=Path("bpti-equil.pdb"))
@click.option("--n-cycles", default=100)
@click.option("--trials-per-cycle", default=1000)
@click.option("--md-steps", default=1000)
@click.option("--batch-size", default=64)
@click.option("--seed", default=0)
@click.option("--empty-sphere", is_flag=True, help="Switch off every water in the sphere first.")
@click.option("--grand-haar", is_flag=True, help="Give grand uniform insertion rotations.")
@click.option("--platform", default="CUDA")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(
    which,
    mode,
    pdb,
    n_cycles,
    trials_per_cycle,
    md_steps,
    batch_size,
    seed,
    empty_sphere,
    grand_haar,
    platform,
    out,
):
    out.mkdir(parents=True, exist_ok=True)
    if not pdb.exists():
        urllib.request.urlretrieve(PDB_URL, pdb)
    np.random.seed(seed)  # grand draws from numpy's global generator
    src = app.PDBFile(str(pdb))
    top, pos, ghosts = add_ghost_waters(src.topology, src.positions, 15, seed=seed)
    with (out / "topology.pdb").open("w") as f:
        app.PDBFile.writeFile(top, pos, f, keepIds=True)
    system = app.ForceField("amber14-all.xml", "amber14/tip3p.xml").createSystem(
        top,
        nonbondedMethod=app.PME,
        nonbondedCutoff=12 * unit.angstrom,
        switchDistance=10 * unit.angstrom,
        constraints=app.HBonds,
    )
    if which == "grand":
        if grand is None:
            raise click.ClickException("grand is not installed")
        sampler = grand.samplers.StandardGCMCSphereSampler(
            system=system,
            topology=top,
            temperature=TEMPERATURE,
            referenceAtoms=REF_ATOMS,
            sphereRadius=RADIUS_A * unit.angstrom,
            ghostFile=str(out / "ghosts.txt"),
            log=str(out / "gcmc.log"),
            overwrite=True,
        )
        if grand_haar:  # Rotation.random draws from numpy's global generator, seeded above
            grand.samplers.random_rotation_matrix = lambda: Rotation.random().as_matrix()
        adams, write_ghosts = float(sampler.B), sampler.writeGhostWaterResids
    else:
        device = "cuda" if platform == "CUDA" else "cpu"
        settings = GCMCSettings(
            sphere_radius_a=RADIUS_A, batch_size=batch_size, seed=seed, device=device
        )
        sampler = GCMCSampler(system, top, REF_ATOMS, settings)
        adams, write_ghosts = settings.b, lambda: sampler.write_ghost_line(out / "ghosts.txt")
    integrator = openmm.LangevinMiddleIntegrator(
        TEMPERATURE, 1 / unit.picosecond, 2 * unit.femtosecond
    )
    props = {"Precision": "mixed"} if platform in ("CUDA", "OpenCL") else {}
    sim = app.Simulation(
        top, system, integrator, openmm.Platform.getPlatformByName(platform), props
    )
    sim.context.setPositions(pos)
    sim.context.setPeriodicBoxVectors(*top.getPeriodicBoxVectors())
    sampler.initialise(sim.context, ghosts)
    if empty_sphere:
        sampler.deleteWatersInGCMCSphere() if which == "grand" else sampler.empty_sphere()
    # After initialise: setting velocities before it makes this system go NaN within 300 steps.
    sim.context.setVelocitiesToTemperature(TEMPERATURE, seed)
    run = {
        "sampler": which, "mode": mode, "pdb": str(pdb), "n_cycles": n_cycles,
        "trials_per_cycle": trials_per_cycle, "md_steps": md_steps if mode == "md" else 0,
        "batch_size": batch_size if which == "parallel" else 1, "seed": seed,
        "empty_sphere": empty_sphere, "grand_haar": grand_haar and which == "grand",
        "platform": platform, "adams": adams,
        "n_atoms": system.getNumParticles(), "ghosts": ghosts, "ref_atoms": REF_ATOMS,
        "sphere_radius_a": RADIUS_A, "temperature_k": TEMPERATURE.value_in_unit(unit.kelvin),
    }  # fmt: skip
    (out / "settings.json").write_text(json.dumps(run, indent=2))

    dcd = None
    if mode == "md":
        (out / "ghosts.txt").write_text("")
        dcd_file = (out / "traj.dcd").open("wb")
        dcd = app.DCDFile(dcd_file, top, 0.002, interval=md_steps)  # dt is one MD step (ps)
    gcmc_s = md_s = 0.0
    start = time.perf_counter()
    with (out / "cycles.csv").open("w") as csv:
        csv.write("cycle,N,trials,accepted,wall_s\n")
        for cycle in range(n_cycles):
            if dcd is not None:
                t = time.perf_counter()
                sim.step(md_steps)
                sim.context.getState(
                    getPositions=True
                )  # wait for the GPU, so MD time stays MD time
                md_s += time.perf_counter() - t
            t = time.perf_counter()
            sampler.move(sim.context, trials_per_cycle)
            gcmc_s += time.perf_counter() - t
            if dcd is not None:
                state = sim.context.getState(getPositions=True)
                dcd.writeModel(
                    state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors()
                )
                write_ghosts()
            wall = time.perf_counter() - start
            csv.write(f"{cycle},{sampler.N},{sampler.n_moves},{sampler.n_accepted},{wall:.3f}\n")
            csv.flush()
            click.echo(
                f"cycle {cycle}: N={sampler.N} trials={sampler.n_moves} acc={sampler.n_accepted}"
            )
            np.save(out / "Ns.npy", np.asarray(sampler.Ns, np.int8))
            timing = {
                "gcmc_ms_per_trial": 1000 * gcmc_s / sampler.n_moves,
                "gcmc_s": gcmc_s,
                "md_s": md_s,
                "md_ms_per_step": 1000 * md_s / ((cycle + 1) * md_steps) if dcd else None,
                "wall_s": time.perf_counter() - start,
                "n_moves": sampler.n_moves,
                "n_accepted": sampler.n_accepted,
                "n_stage1_accepted": getattr(sampler, "n_stage1_accepted", None),
                "n_stage2_rejected": getattr(sampler, "n_stage2_rejected", None),
            }
            (out / "timing.json").write_text(json.dumps(timing, indent=2))
    if dcd is not None:
        dcd_file.close()
    click.echo(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
