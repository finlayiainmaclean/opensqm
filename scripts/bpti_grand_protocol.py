#!/usr/bin/env python
"""grand's BPTI equilibration and production, with either sampler, and restraints on the solute.

Stages follow grand's examples/bpti (equil-uvt1, equil-npt, equil-uvt2, prod/bpti.py):

1. uVT: empty the sphere, 10,000 trials, then 100 blocks of 1,000 trials + 5 MD steps.
2. NPT: ghosts removed, 250,000 MD steps at 1 bar, no GCMC.
3. uVT: 15 new ghosts, 500 cycles of 500 MD steps + 200 trials.
4. Production: restraints off, cycles of 1,000 MD steps + 100 trials.

Unlike grand's scripts, stages 1-3 hold the heavy atoms of every residue that is not water
or an ion near their starting positions (``--restraint-k``, kcal/mol/A^2). The run
directory gets uvt1.csv, uvt2.csv, cycles.csv (production), traj.dcd and ghosts.txt
(production, grand's format), topology.pdb, settings.json and timing.json.

    python scripts/bpti_grand_protocol.py --sampler parallel --seed 1 --out runs/p1
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

PDB_URL = "https://raw.githubusercontent.com/essex-lab/grand/9659ed2/examples/bpti/equil/bpti.pdb"
REF_ATOMS = [
    {"name": "CA", "resname": "TYR", "resid": "10"},
    {"name": "CA", "resname": "ASN", "resid": "43"},
]
RADIUS_A, T = 4.2, 298.0 * unit.kelvin
FF = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
IONS = {"NA", "CL", "Na+", "Cl-", "K", "K+", "MG", "CA2"}


def build(top, pos, k, barostat=False):
    """grand's BPTI System, plus heavy-atom restraints (global parameter k_restr) on the solute."""
    system = FF.createSystem(
        top,
        nonbondedMethod=app.PME,
        nonbondedCutoff=12 * unit.angstrom,
        switchDistance=10 * unit.angstrom,
        constraints=app.HBonds,
    )
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setUseDispersionCorrection(False)  # as grand's equil-npt; its samplers do too
    restraint = openmm.CustomExternalForce("0.5*k_restr*periodicdistance(x,y,z,x0,y0,z0)^2")
    restraint.addGlobalParameter("k_restr", k * 4.184 * 100)  # kcal/mol/A^2 -> kJ/mol/nm^2
    for name in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    xyz = pos.value_in_unit(unit.nanometer)
    for atom in top.atoms():
        solute = atom.residue.name not in IONS | {"HOH"}
        if solute and atom.element is not None and atom.element.symbol != "H":
            restraint.addParticle(atom.index, list(xyz[atom.index]))
    system.addForce(restraint)
    if barostat:
        system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, T, 25))
    return system


def simulate(top, system, pos, box, platform, seed):
    """A Simulation with grand's integrator settings (BAOAB as LangevinMiddle, 2 fs)."""
    integrator = openmm.LangevinMiddleIntegrator(T, 1 / unit.picosecond, 2 * unit.femtosecond)
    integrator.setRandomNumberSeed(seed)
    props = {"Precision": "mixed"} if platform in ("CUDA", "OpenCL") else {}
    sim = app.Simulation(
        top, system, integrator, openmm.Platform.getPlatformByName(platform), props
    )
    sim.context.setPositions(pos)
    sim.context.setPeriodicBoxVectors(*box)
    return sim


def gcmc(which, system, top, out, stage, seed, batch_size, platform):
    """A sampler of either kind, built on ``system`` before its Context exists."""
    if which == "grand":
        g = grand.samplers.StandardGCMCSphereSampler(
            system=system, topology=top, temperature=T, referenceAtoms=REF_ATOMS,
            sphereRadius=RADIUS_A * unit.angstrom, ghostFile=str(out / f"{stage}-ghosts.txt"),
            log=str(out / f"{stage}.log"), overwrite=True,
        )  # fmt: skip
        g.ghost_resids_now = lambda: g.getWaterStatusResids(0)
        return g
    settings = GCMCSettings(
        sphere_radius_a=RADIUS_A,
        batch_size=batch_size,
        seed=seed,
        device="cuda" if platform == "CUDA" else "cpu",
    )
    s = GCMCSampler(system, top, REF_ATOMS, settings)
    s.ghost_resids_now = lambda: s.ghost_resids
    return s


def without(top, pos, resids):
    """Topology and positions with the given residues removed (grand's remove_ghosts)."""
    modeller = app.Modeller(top, pos)
    residues = list(top.residues())
    modeller.delete([residues[i] for i in resids])
    return modeller.topology, modeller.positions


def log_row(f, i, sampler, t0):
    f.write(
        f"{i},{sampler.N},{sampler.n_moves},{sampler.n_accepted},{time.perf_counter() - t0:.3f}\n"
    )
    f.flush()


@click.command()
@click.option("--sampler", "which", type=click.Choice(["grand", "parallel"]), required=True)
@click.option("--seed", default=1)
@click.option("--batch-size", default=100)
@click.option("--trials-per-cycle", default=100, help="Production GCMC trials per 2 ps.")
@click.option("--prod-cycles", default=1000, help="Production cycles of 2 ps.")
@click.option("--restraint-k", default=5.0, help="kcal/mol/A^2 on solute heavy atoms, stages 1-3.")
@click.option("--pdb", type=click.Path(path_type=Path), default=Path("bpti.pdb"))
@click.option("--scale", default=1.0, help="Shrink every stage, for a smoke test.")
@click.option("--platform", default="CUDA")
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(
    which, seed, batch_size, trials_per_cycle, prod_cycles, restraint_k, pdb, scale, platform, out
):
    if which == "grand" and grand is None:
        raise click.ClickException("grand is not installed")
    out.mkdir(parents=True, exist_ok=True)
    if not pdb.exists():
        urllib.request.urlretrieve(PDB_URL, pdb)
    np.random.seed(seed)  # grand draws from numpy's global generator
    if which == "grand":  # uniform insertion rotations (grand's are not uniform over SO(3))
        grand.samplers.random_rotation_matrix = lambda: Rotation.random().as_matrix()
    src = app.PDBFile(str(pdb))
    box = src.topology.getPeriodicBoxVectors()
    timing = {}

    # 1. uVT on a near-static structure, from an empty sphere.
    t0 = time.perf_counter()
    top, pos, ghosts = add_ghost_waters(src.topology, src.positions, 15, seed=seed)
    system = build(top, pos, restraint_k)
    sampler = gcmc(which, system, top, out, "uvt1", seed, batch_size, platform)
    sim = simulate(top, system, pos, box, platform, seed)
    sampler.initialise(sim.context, ghosts)
    sampler.deleteWatersInGCMCSphere() if which == "grand" else sampler.empty_sphere()
    sim.context.setVelocitiesToTemperature(T, seed)  # after initialise: before it goes NaN
    with (out / "uvt1.csv").open("w") as f:
        f.write("block,N,trials,accepted,wall_s\n")
        for i, n in enumerate([int(10000 * scale)] + [1000] * int(100 * scale)):
            sampler.move(sim.context, n)
            if i:
                sim.step(5)
            log_row(f, i, sampler, t0)
    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    top, pos = without(top, state.getPositions(), sampler.ghost_resids_now())
    timing["uvt1_s"] = time.perf_counter() - t0

    # 2. NPT MD, no GCMC.
    t0 = time.perf_counter()
    sim = simulate(top, build(top, pos, restraint_k, barostat=True), pos, box, platform, seed)
    sim.context.setVelocitiesToTemperature(T, seed)
    sim.step(int(250000 * scale))
    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    pos, box = state.getPositions(), state.getPeriodicBoxVectors()
    top.setPeriodicBoxVectors(box)
    timing["npt_s"] = time.perf_counter() - t0

    # 3. uVT with MD, then 4. production with the restraints off, in one Context.
    t0 = time.perf_counter()
    top, pos, ghosts = add_ghost_waters(top, pos, 15, seed=seed + 1000)
    with (out / "topology.pdb").open("w") as f:
        app.PDBFile.writeFile(top, pos, f, keepIds=True)
    system = build(top, pos, restraint_k)
    sampler = gcmc(which, system, top, out, "prod", seed, batch_size, platform)
    sim = simulate(top, system, pos, box, platform, seed)
    sampler.initialise(sim.context, ghosts)
    sim.context.setVelocitiesToTemperature(T, seed)
    with (out / "uvt2.csv").open("w") as f:
        f.write("cycle,N,trials,accepted,wall_s\n")
        for i in range(int(500 * scale)):
            sim.step(500)
            sampler.move(sim.context, 200)
            log_row(f, i, sampler, t0)
    timing["uvt2_s"] = time.perf_counter() - t0

    sim.context.setParameter("k_restr", 0.0)
    if which == "grand":
        sampler.reset()
    else:
        sampler.Ns, sampler.n_moves, sampler.n_accepted = [], 0, 0
    (out / "ghosts.txt").write_text("")
    write_ghosts = sampler.writeGhostWaterResids if which == "grand" else (
        lambda: sampler.write_ghost_line(out / "ghosts.txt")
    )  # fmt: skip
    if which == "grand":
        sampler.ghost_file = str(out / "ghosts.txt")
    settings = {
        "sampler": which, "seed": seed, "batch_size": batch_size,
        "trials_per_cycle": trials_per_cycle,
        "md_steps": 1000, "prod_cycles": prod_cycles, "restraint_k_kcal_A2": restraint_k,
        "sphere_radius_a": RADIUS_A, "ref_atoms": REF_ATOMS, "grand_haar": which == "grand",
        "mode": "md", "empty_sphere": True, "equil_trials": 110000, "protocol": "grand-restrained",
    }  # fmt: skip
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    t0, gcmc_s = time.perf_counter(), 0.0
    with (out / "traj.dcd").open("wb") as dcd_file, (out / "cycles.csv").open("w") as f:
        dcd = app.DCDFile(dcd_file, top, 0.002, interval=1000)
        f.write("cycle,N,trials,accepted,wall_s\n")
        for i in range(int(prod_cycles * scale)):
            sim.step(1000)
            t = time.perf_counter()
            sampler.move(sim.context, trials_per_cycle)
            gcmc_s += time.perf_counter() - t
            state = sim.context.getState(getPositions=True)
            dcd.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())
            write_ghosts()
            log_row(f, i, sampler, t0)
            timing.update(prod_s=time.perf_counter() - t0, prod_gcmc_s=gcmc_s)
            timing.update(
                gcmc_ms_per_trial=1000 * gcmc_s / sampler.n_moves, n_moves=sampler.n_moves
            )
            timing.update(
                n_accepted=sampler.n_accepted, gcmc_s=gcmc_s, md_s=timing["prod_s"] - gcmc_s
            )
            (out / "timing.json").write_text(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
