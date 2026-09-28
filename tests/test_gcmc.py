"""Tests for batched GCMC: the stage-1 energy, the ideal-gas limit and context consistency."""

import math

import numpy as np
import openmm
import pytest
from openmm import app, unit
from pymbar import timeseries

import opensqm.gcmc.sampler
from opensqm.gcmc import GCMCSampler, GCMCSettings, water_interaction_energy

TIP3P = np.array([[-0.834, 0.315061, 0.636386], [0.417, 0.1, 0.05], [0.417, 0.1, 0.05]])
GEOMETRY = 0.09572 * np.array([[0, 0, 0], [1, 0, 0], [math.cos(1.8242), math.sin(1.8242), 0]])
NONE = -(10**9)  # own_start of an insertion


def _context(system, platform="Reference"):
    platform = openmm.Platform.getPlatformByName(platform)
    return openmm.Context(system, openmm.VerletIntegrator(0.001), platform)


def _energy(ctx):
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _system(box, n_waters, n_ions, method, charged=True):
    """Waters then LJ ions on a jittered grid in ``box``, some shifted by a box vector."""
    rng = np.random.default_rng(7)
    frac = (np.indices((4, 4, 4)).reshape(3, -1).T + 0.5) / 4
    frac = frac[rng.permutation(64)[: n_waters + n_ions]] + rng.uniform(-0.03, 0.03, (1, 3))
    centres = (frac + rng.integers(-1, 2, frac.shape) * (rng.random((len(frac), 1)) < 0.3)) @ box
    system, top = openmm.System(), app.Topology()
    system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*v) for v in box])
    top.setPeriodicBoxVectors(box * unit.nanometer)
    chain = top.addChain()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(method)
    nb.setCutoffDistance(1.0)
    nb.setReactionFieldDielectric(78.3)
    nb.setUseDispersionCorrection(False)
    pos = []
    for w in range(n_waters):
        res = top.addResidue("HOH", chain)
        rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        for name, p, g in zip(("O", "H1", "H2"), TIP3P, GEOMETRY, strict=True):
            top.addAtom(name, app.element.oxygen if name == "O" else app.element.hydrogen, res)
            system.addParticle(16.0)
            nb.addParticle(*(p * [charged, 1, charged]))
            pos.append(centres[w] + rot @ g)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            nb.addException(3 * w + i, 3 * w + j, 0.0, 0.1, 0.0)
    for k in range(n_ions):
        top.addAtom("NA", app.element.sodium, top.addResidue("ION", chain))
        system.addParticle(23.0)
        nb.addParticle(rng.choice([-0.5, 0.5]) * charged, 0.25 + 0.1 * rng.random(), 0.4)
        pos.append(centres[n_waters + k])
    system.addForce(nb)
    return system, top, np.array(pos)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])  # the sampler uses float32
@pytest.mark.parametrize(
    "box",
    [np.diag([2.4, 2.4, 2.4]), np.array([[2.4, 0, 0], [0.8, 2.3, 0], [-0.9, 0.7, 2.2]])],
)
def test_stage1_energy_matches_openmm(box, dtype):
    system, _, pos = _system(box, 6, 20, openmm.NonbondedForce.CutoffPeriodic)
    nb = system.getForce(0)
    params = np.array([[x._value for x in nb.getParticleParameters(i)] for i in range(len(pos))])
    ctx = _context(system)

    def openmm_interaction(xyz, on, w):
        """E(all on) - E(all on but water w) - E(water w alone)."""
        total = 0.0
        for sign, mask in ((1, on), (-1, on & ~w), (-1, w)):
            for i, (q, s, e) in enumerate(params):
                nb.setParticleParameters(i, q * mask[i], s, e * mask[i])
            nb.updateParametersInContext(ctx)
            ctx.setPositions(xyz)
            total += sign * _energy(ctx)
        return total

    def stage1(xyz, w, own_start):
        xyz = xyz.astype(dtype)
        args = (TIP3P.astype(dtype), xyz, params.astype(dtype), real, np.array([own_start]))
        return water_interaction_energy(np, xyz[w][None], *args, box.astype(dtype), 1.0, 78.3)[0]

    tol = {"rel": 1e-7, "abs": 1e-6} if dtype == np.float64 else {"abs": 1e-2}

    water = [np.arange(len(pos)) // 3 == w for w in range(6)]
    real = ~water[0]  # water 0 is the ghost
    rng = np.random.default_rng(1)
    for w in (1, 2, 3):  # deletion: a real water, its own atoms left out
        expected = openmm_interaction(pos, real, water[w])
        assert stage1(pos, water[w], 3 * w) == pytest.approx(expected, **tol)
    for _ in range(3):  # insertion: the ghost at a random place and orientation
        new = pos.copy()
        new[water[0]] = rng.random(3) @ box + GEOMETRY @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
        expected = openmm_interaction(new, np.ones_like(real), water[0])
        assert stage1(new, water[0], NONE) == pytest.approx(expected, **tol)


def _gcmc(charged, n_waters, n_ghosts, platform="Reference", **settings):
    """A gas of waters around one LJ particle, which centres an 8 A GCMC sphere."""
    box = np.diag([2.5, 2.5, 2.5])
    system, top, pos = _system(box, n_waters, 1, openmm.NonbondedForce.PME, charged)
    pos[-1] = box.diagonal() / 2
    settings = GCMCSettings(sphere_radius_a=8.0, device="cpu", **settings)
    sampler = GCMCSampler(system, top, [len(pos) - 1], settings)
    ctx = _context(system, platform)
    ctx.setPositions(pos)
    sampler.initialise(ctx, list(range(n_ghosts)))
    return sampler, ctx


@pytest.mark.parametrize("perturbed", [False, True])
@pytest.mark.parametrize("batch_size", [1, 16])
def test_ideal_gas_is_poisson(batch_size, perturbed, monkeypatch):
    """With ``perturbed``, stage 1 sees a bounded fake energy, so stage 2 must correct it."""
    assert GCMCSettings(sphere_radius_a=4.2).b == pytest.approx(-7.9589, abs=1e-3)  # grand, BPTI
    if perturbed:
        fake = lambda xp, sites, *a: 2 * np.sin(20 * sites[:, 0, 0]) + 3  # noqa: E731
        monkeypatch.setattr(opensqm.gcmc.sampler, "water_interaction_energy", fake)
    lam = 2.5
    sampler, ctx = _gcmc(False, 16, 16, "CPU", adams=math.log(lam), batch_size=batch_size)
    sampler.move(ctx, 8000)
    n = np.array(sampler.Ns[500:], float)
    n_eff = len(n) / timeseries.statistical_inefficiency(n)
    assert (sampler.n_stage2_rejected > 0) == perturbed
    assert abs(n.mean() - lam) < 4 * math.sqrt(lam / n_eff)
    assert abs(n.var() - lam) < 4 * math.sqrt((lam + 2 * lam**2) / n_eff)


def test_context_matches_statuses():
    sampler, ctx = _gcmc(True, 20, 12, adams=-3.0, batch_size=8)
    for _ in range(3):
        sampler.move(ctx, 150)
    assert sampler.n_accepted > 0
    assert sampler.n_stage2_rejected > 0
    assert len(sampler.Ns) == sampler.n_moves == 450
    assert sampler.N == (sampler.status == 1).sum()
    for atoms, status in zip(sampler.water_atoms, sampler.status, strict=True):
        for a, q in zip(atoms, sampler.wq, strict=True):
            charge, _, eps = sampler.nb.getParticleParameters(int(a))
            assert charge._value == pytest.approx(q * (status > 0))
            assert eps._value == 0.0
            assert sampler.custom.getParticleParameters(int(a))[2] == float(status > 0)
    # A context built fresh from the System reads the parameters the sampler set.
    rebuilt = _context(ctx.getSystem())
    rebuilt.setPositions(ctx.getState(getPositions=True).getPositions())
    assert sampler.energy == pytest.approx(_energy(ctx), rel=1e-9, abs=1e-6)
    assert sampler.energy == pytest.approx(_energy(rebuilt), rel=1e-9, abs=1e-6)
