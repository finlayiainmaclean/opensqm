"""Grand canonical Monte Carlo of water in a sphere, with trials evaluated in batches.

The ensemble, Hamiltonian and trial moves are those of grand's
``StandardGCMCSphereSampler`` (Ross et al., JCTC 2020, doi:10.1021/acs.jctc.0c00660).
A move proposes a batch of insertion and deletion trials from one state. Stage 1
screens the whole batch at once with a cheap reaction-field energy (numpy or
cupy). Stage 2 evaluates each survivor with the exact OpenMM energy and corrects
for the stage-1 approximation (Gelb, J. Chem. Phys. 2003, doi:10.1063/1.1563597).
The first stage-2 acceptance ends the batch, and the next batch starts from the
new state, so the chain is the same as a one-trial-at-a-time two-stage (delayed
acceptance) chain. That chain has grand's stationary distribution, but its
acceptance per trial is never more than grand's.
"""

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import openmm
from openmm import app, unit
from pydantic import BaseModel

from opensqm.gcmc.energy import min_image, water_interaction_energy

try:
    import cupy as cp
except ImportError:
    cp = None

KJ = unit.kilojoule_per_mole
NM = unit.nanometer
# Softcore sterics expression and parameters from grand (Samways, Melling; MIT).
SOFTCORE = (
    "U; U = (lambda^soft_a) * 4 * epsilon * x * (x-1.0); x = (sigma/reff)^6;"
    "reff = sigma*((soft_alpha*(1.0-lambda)^soft_b + (r/sigma)^soft_c))^(1/soft_c);"
    "sigma = 0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2); lambda = lambda1*lambda2"
)


class GCMCSettings(BaseModel):
    """Settings for one ``GCMCSampler``.

    Give ``sphere_radius_a`` (A); the other defaults are grand's. ``adams`` sets
    the Adams value B directly; if it is None, B is mu'/kT + ln(V_sphere/V_standard).
    ``batch_size`` trials are proposed from one state and screened together on
    ``device``: "cuda" uses cupy, "cpu" uses numpy. ``seed`` seeds the one numpy
    Generator that draws every random number of the sampler. ``sphere_centre_nm``
    fixes the sphere in space (grand's ``sphereCentre``); then ``reference_atoms``
    may be empty.
    """

    sphere_radius_a: float
    temperature_k: float = 298.0
    excess_chemical_potential_kcal: float = -6.09
    standard_volume_a3: float = 30.345
    adams: float | None = None
    batch_size: int = 64
    seed: int = 0
    device: Literal["cuda", "cpu"] = "cuda"
    rf_dielectric: float = 78.3
    sphere_centre_nm: tuple[float, float, float] | None = None

    @property
    def kt(self) -> float:
        """Thermal energy kT in kJ/mol."""
        return (unit.MOLAR_GAS_CONSTANT_R * self.temperature_k * unit.kelvin).value_in_unit(KJ)

    @property
    def b(self) -> float:
        """Adams value B."""
        if self.adams is not None:
            return self.adams
        mu = (self.excess_chemical_potential_kcal * unit.kilocalorie_per_mole).value_in_unit(KJ)
        volume = 4 / 3 * math.pi * self.sphere_radius_a**3
        return mu / self.kt + math.log(volume / self.standard_volume_a3)


def add_ghost_waters(
    topology: app.Topology, positions: Any, n: int, seed: int = 0
) -> tuple[app.Topology, Any, list[int]]:
    """Append ``n`` copies of the first HOH residue at random points in the box.

    Adapted from grand.utils.add_ghosts. Returns the new topology, the new
    positions and the residue indices of the added waters, to pass to
    ``GCMCSampler.initialise``.
    """
    water = next(r for r in topology.residues() if r.name == "HOH")
    xyz = np.array(positions.value_in_unit(NM))[[a.index for a in water.atoms()]]
    template = app.Topology()
    residue = template.addResidue("HOH", template.addChain())
    atoms = {a: template.addAtom(a.name, a.element, residue) for a in water.atoms()}
    for a1, a2 in water.bonds():
        template.addBond(atoms[a1], atoms[a2])
    modeller = app.Modeller(topology, positions)
    box = np.array(topology.getPeriodicBoxVectors().value_in_unit(NM))
    rng = np.random.default_rng(seed)
    first = topology.getNumResidues()
    for _ in range(n):
        modeller.add(template, (xyz - xyz[0] + rng.random(3) @ box) * NM)
    return modeller.topology, modeller.positions, list(range(first, first + n))


class GCMCSampler:
    """Water GCMC in a sphere, with each batch of trials screened in one call.

    Build it on the ``System`` before the ``Context`` exists: it moves Lennard-Jones
    into a softcore ``CustomNonbondedForce`` as grand does. Then call
    ``initialise(context, ghost_resids)``, set velocities, and alternate MD with
    ``move(context, n)``. ``reference_atoms`` are atom indices or grand-style dicts
    (``name``, ``resname``, optional ``resid`` and ``chain``); the sphere is centred
    on their centre of geometry. Water status: 0 ghost, 1 real in the sphere,
    2 real outside it.
    """

    def __init__(
        self,
        system: openmm.System,
        topology: app.Topology,
        reference_atoms: list[int] | list[dict],
        settings: GCMCSettings,
    ) -> None:
        if settings.device == "cuda" and cp is None:
            raise ImportError("device 'cuda' needs cupy")
        self.xp: Any = cp if settings.device == "cuda" else np
        self.settings, self.kt, self.b = settings, settings.kt, settings.b
        self.rng = np.random.default_rng(settings.seed)
        self.context: openmm.Context | None = None
        forces = system.getForces()
        if any("Barostat" in type(f).__name__ for f in forces):
            raise ValueError("GCMC needs constant volume; remove the barostat")
        nb = next((f for f in forces if isinstance(f, openmm.NonbondedForce)), None)
        if nb is None:
            raise ValueError("GCMC needs a NonbondedForce")
        self.nb = nb
        if self.nb.getNonbondedMethod() != openmm.NonbondedForce.PME:
            raise ValueError("GCMC needs PME electrostatics")
        residues = list(topology.residues())
        self.waters = [r.index for r in residues if r.name == "HOH"]
        self.water_atoms = np.array([[a.index for a in residues[w].atoms()] for w in self.waters])
        if (np.diff(self.water_atoms, axis=1) != 1).any():
            raise ValueError("the atoms of each water must be contiguous")
        self.ref = [
            i for r in reference_atoms
            for i in (_find_atoms(topology, r) if isinstance(r, dict) else [int(r)])
        ]  # fmt: skip
        if not self.ref and settings.sphere_centre_nm is None:
            raise ValueError("give reference_atoms or settings.sphere_centre_nm")

        # customiseForces from grand (Samways, Melling; MIT)
        custom = openmm.CustomNonbondedForce(SOFTCORE)
        for name in ("sigma", "epsilon", "lambda"):
            custom.addPerParticleParameter(name)
        custom.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        custom.setUseSwitchingFunction(self.nb.getUseSwitchingFunction())
        custom.setCutoffDistance(self.nb.getCutoffDistance())
        custom.setSwitchingDistance(self.nb.getSwitchingDistance())
        self.nb.setUseDispersionCorrection(False)
        custom.setUseLongRangeCorrection(False)
        for name, value in (("soft_alpha", 0.5), ("soft_a", 1), ("soft_b", 1), ("soft_c", 6)):
            custom.addGlobalParameter(name, value)
        params = np.zeros((system.getNumParticles(), 3))
        for i in range(system.getNumParticles()):
            q, sig, eps = (
                x.value_in_unit_system(unit.md_unit_system)
                for x in self.nb.getParticleParameters(i)
            )
            sig = 0.1 if np.isclose(sig, 0.0) else sig
            params[i] = q, sig, eps
            custom.addParticle([sig, eps, 1.0])
            self.nb.setParticleParameters(i, q, sig, 0.0)
        water_set = set(self.water_atoms.ravel().tolist())
        for k in range(self.nb.getNumExceptions()):
            i, j, _, _, eps = self.nb.getExceptionParameters(k)
            if eps.value_in_unit(KJ) > 0 and (i in water_set or j in water_set):
                raise ValueError(f"non-zero exception between atoms {i} and {j} involves a water")
            custom.addExclusion(i, j)
        system.addForce(custom)
        self.custom = custom

        self.wq, self.wsig, self.weps = params[self.water_atoms[0]].T
        params[self.water_atoms] = params[self.water_atoms[0]]
        self.cutoff = self.nb.getCutoffDistance().value_in_unit(NM)
        self._site = self.xp.asarray(params[self.water_atoms[0]], np.float32)
        self._atom = self.xp.asarray(params, np.float32)
        self.status = np.ones(len(self.waters), int)
        self.N, self.Ns = 0, []
        self.n_moves = self.n_accepted = self.n_stage1_accepted = self.n_stage2_rejected = 0

    @property
    def ghost_resids(self) -> list[int]:
        """Residue indices of the ghost waters."""
        return [self.waters[i] for i in np.flatnonzero(self.status == 0)]

    def write_ghost_line(self, path: str | Path) -> None:
        """Append the ghost residue indices to ``path`` in grand's ghost-file format."""
        with Path(path).open("a") as f:
            f.write(",".join(map(str, self.ghost_resids)) + "\n")

    def initialise(self, context: openmm.Context, ghost_resids: list[int]) -> None:
        """Switch off the ghost waters and read the water template and statuses."""
        self.context = context
        ghosts = set(ghost_resids)
        for i, w in enumerate(self.waters):
            if w in ghosts:
                self._set_water(i, on=False, update=False)
                self.status[i] = 0
        self._update_context()
        state = context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(NM)
        if self.settings.sphere_radius_a / 10 > 0.5 * box.diagonal().min():
            raise ValueError("GCMC sphere radius cannot be larger than half a box length")
        self.energy = state.getPotentialEnergy().value_in_unit(KJ)
        pos = state.getPositions(asNumpy=True).value_in_unit(NM)
        self.template = pos[self.water_atoms[0]] - pos[self.water_atoms[0, 0]]
        self._refresh(state)

    def empty_sphere(self) -> None:
        """Switch off every water in the sphere (grand's deleteWatersInGCMCSphere)."""
        for i in np.flatnonzero(self.status == 1):
            self._set_water(i, on=False, update=False)
        self.status[self.status == 1] = 0
        self.N = 0
        self._update_context()

    def move(self, context: openmm.Context, n: int) -> None:
        """Do exactly ``n`` insertion or deletion trials, with grand's semantics."""
        self.context = context
        state = context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
        self.energy = state.getPotentialEnergy().value_in_unit(KJ)
        self._refresh(state)
        xp, s = self.xp, self.settings
        radius = s.sphere_radius_a / 10
        done = 0
        while done < n:
            b = min(s.batch_size, n - done)
            insert = self.rng.random(b) < 0.5
            u1, u2 = np.log(1 - self.rng.random((2, b)))
            direction = self.rng.normal(size=(b, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            direction *= radius * self.rng.random((b, 1)) ** (1 / 3)
            # QR of a Gaussian matrix, with signs fixed, is a uniform rotation or reflection.
            rot, r = np.linalg.qr(self.rng.normal(size=(b, 3, 3)))
            rot *= np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
            rot *= np.linalg.det(rot)[:, None, None]  # a reflection times -1 is a rotation
            in_sphere = np.flatnonzero(self.status == 1)
            pick = in_sphere[self.rng.integers(self.N, size=b)] if self.N else np.zeros(b, int)
            new = self.centre + direction[:, None] + np.einsum("bij,sj->bsi", rot, self.template)
            sites = np.where(insert[:, None, None], new, self.pos[self.water_atoms[pick]])
            own = np.where(insert, -(10**9), self.water_atoms[pick, 0])
            e = water_interaction_energy(
                xp, xp.asarray(sites, np.float32), self._site, self._xyz, self._atom, self._real,
                xp.asarray(own), self._box, self.cutoff, s.rf_dielectric,
            )  # fmt: skip
            e = (cp.asnumpy(e) if xp is cp else e).astype(float)
            du = np.where(insert, e, -e)
            log_pre = np.where(
                insert, self.b - math.log(self.N + 1), -self.b + math.log(max(self.N, 1))
            )
            stage1 = np.isfinite(e) & (u1 < log_pre - du / self.kt) & (insert | (self.N > 0))
            consumed, n_before = b, self.N
            for k in np.flatnonzero(stage1):
                self.n_stage1_accepted += 1
                if self._stage2(bool(insert[k]), int(pick[k]), sites[k], du[k], u2[k]):
                    consumed = int(k) + 1
                    self.n_accepted += 1
                    break
                self.n_stage2_rejected += 1
            self.Ns.extend([n_before] * (consumed - 1) + [self.N])
            self.n_moves += consumed
            done += consumed

    def _stage2(self, insert: bool, pick: int, sites: np.ndarray, du: float, log_u: float) -> bool:
        """Apply one trial in OpenMM; keep it if the Gelb correction accepts it, else revert."""
        context, kt = self.context, self.kt
        assert context is not None
        if insert:
            ghosts = np.flatnonzero(self.status == 0)
            if len(ghosts) == 0:
                raise RuntimeError("no ghost water left for an insertion; add more ghost waters")
            # ponytail: the first ghost, where grand picks a random one; the same, since ghosts
            # do not interact and the new sites replace the ghost's positions.
            pick = int(ghosts[0])
            new_pos = self.pos.copy()
            new_pos[self.water_atoms[pick]] = sites
            self._set_water(pick, on=True)
            context.setPositions(new_pos * NM)
        else:
            self._set_water(pick, on=False)
        e_new = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
        if not log_u < -((e_new - self.energy) - du) / kt:
            self._set_water(pick, on=not insert)
            if insert:
                context.setPositions(self.pos * NM)
            return False
        atoms = self.water_atoms[pick]
        if insert:
            self.pos = new_pos
            self._xyz[atoms[0] : atoms[-1] + 1] = self.xp.asarray(sites, np.float32)
        self.status[pick] = 1 if insert else 0
        self._real[atoms[0] : atoms[-1] + 1] = insert
        self.N += 1 if insert else -1
        self.energy = e_new
        return True

    def _set_water(self, i: int, on: bool, update: bool = True) -> None:
        """Give water ``i`` its template charges and lambda 1 (on) or zero both (off)."""
        for a, q, sig, eps in zip(self.water_atoms[i], self.wq, self.wsig, self.weps, strict=True):
            self.nb.setParticleParameters(int(a), q * on, sig, 0.0)
            self.custom.setParticleParameters(int(a), [sig, eps, float(on)])
        if update:
            self._update_context()

    def _update_context(self) -> None:
        self.nb.updateParametersInContext(self.context)
        self.custom.updateParametersInContext(self.context)

    def _refresh(self, state: openmm.State) -> None:
        """Read positions, sphere centre and water statuses, and copy them to the device."""
        self.pos = state.getPositions(asNumpy=True).value_in_unit(NM)
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(NM)
        if self.settings.sphere_centre_nm is not None:
            self.centre = np.array(self.settings.sphere_centre_nm)
        else:
            ref = self.pos[self.ref]
            self.centre = ref[0] + min_image(np, ref - ref[0], box).mean(axis=0)
        oxygen = min_image(np, self.pos[self.water_atoms[:, 0]] - self.centre, box)
        inside = np.linalg.norm(oxygen, axis=1) <= self.settings.sphere_radius_a / 10
        self.status = np.where(self.status == 0, 0, np.where(inside, 1, 2))
        self.N = int((self.status == 1).sum())
        real = np.ones(len(self.pos), bool)
        real[self.water_atoms[self.status == 0].ravel()] = False
        self._xyz = self.xp.asarray(self.pos, np.float32)
        self._real = self.xp.asarray(real)
        self._box = self.xp.asarray(box, np.float32)


def _find_atoms(topology: app.Topology, ref: dict) -> list[int]:
    """Return the index of every atom a grand-style reference dict matches, as grand does."""
    found = [
        atom.index
        for atom in topology.atoms()
        if (atom.name, atom.residue.name) == (ref["name"], ref["resname"])
        and ref.get("resid", atom.residue.id) == atom.residue.id
        and ref.get("chain", atom.residue.chain.index) == atom.residue.chain.index
    ]
    if not found:
        raise ValueError(f"reference atom {ref} not found")
    return found
