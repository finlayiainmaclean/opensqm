"""Water-swap Monte Carlo with two persistent ghost slots.

Implements the NCMC water-swap algorithm described by Ben-Shalom & co
(J. Chem. Theory Comput. 2019, 15, 2684) using a *two-ghost* design:
two designated water residues - ``ghost1`` (always on outside attempts)
and ``ghost2`` (always off outside attempts) - act as persistent swap
buffers. The number of *interacting* waters in the box is therefore
constant throughout the whole simulation (``N_solvent - 1``), so the
physical observable is unchanged by introducing the mover and the
chain is exactly reversible.

Algorithm (single attempt)
--------------------------
*Insert ("in") move - bulk to active-site.*

1. ``ghost2`` (off) is translated to a uniform random point inside
   the active-site sphere. Because ``ghost2`` carries zero charge and
   zero :math:`\\epsilon` between moves, this translation costs zero
   energy regardless of where it lands.
2. A single linear-alchemy switch is run with ``lambda_water_swap``
   driving simultaneously: ``ghost1`` decouples from ``1 -> 0`` while
   ``ghost2`` couples up from ``0 -> 1``. The two scaling laws share
   one global parameter so the perturbation is genuinely
   simultaneous, with MD propagation interleaved between perturbation
   jumps.
3. On acceptance the positions (and velocities) of the two ghost
   atoms are swapped and ``lambda_water_swap`` is reset to ``0``.
   This restores the invariant "``ghost1`` is the on-slot, ``ghost2``
   is the off-slot", with ``ghost1`` now sitting at the active-site
   point chosen in step 1.

*Delete ("out") move - active-site to bulk.*

1. A random *real* water ``W`` is picked from the active site
   (uniform over current pocket waters, excluding the two ghosts).
   ``W`` and ``ghost1`` swap atomic positions and velocities, which
   is a pure relabelling because both are on, so the energy is
   unchanged. After the swap ``ghost1`` is the one inside the
   pocket and ``W`` sits where ``ghost1`` was in bulk.
2. ``ghost2`` (off) is translated to a uniform random point in bulk.
3. NCMC switching runs exactly as for the insert move
   (``lambda_water_swap``: ``0 -> 1``, ``ghost1`` decouples and
   ``ghost2`` couples).
4. On acceptance the ghost atoms are swapped again, restoring
   ``ghost1`` as the on-slot. End state: ``ghost1`` in bulk (at the
   sampled bulk point), ``ghost2`` off and sitting at ``W``'s old
   pocket position, ``W`` itself sitting in bulk where ``ghost1``
   used to be.

Acceptance
----------
The Metropolis-Hastings test uses the paper's formula

* in:  ``P = min(1, N_B(t0) * V_act / (N_act(tEnd) * V_B) * exp(-beta * w))``
* out: ``P = min(1, N_act(t0) * V_B / (N_B(tEnd) * V_act) * exp(-beta * w))``

where the water counts are over *interacting* waters only (so
``ghost2`` is excluded at ``t0`` and ``ghost1`` is excluded at
``tEnd``).

Composability with ConstantPH
-----------------------------
This module installs its alchemy on the explicit production system's
:class:`~openmm.NonbondedForce` only - it doesn't touch the implicit
or relaxation systems. The constant-pH state-swap machinery never
modifies water-atom parameters (it only walks over titratable residue
atoms), so the ghost offsets coexist peacefully with both REST2 and
the per-state nonbonded reassignment that
:meth:`opensqm.cph.constantph.ConstantPH._applyStateToContext`
performs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from openmm import CustomCentroidBondForce, NonbondedForce, unit
from openmm.unit import MOLAR_GAS_CONSTANT_R

if TYPE_CHECKING:
    from openmm.app import Simulation, Topology


class WaterSwapSwitchFailed(RuntimeError):
    """The NCMC switching protocol produced a non-finite energy or
    blew up the integrator. :meth:`WaterSwapMC.attempt` catches this,
    restores the pre-attempt snapshot, and counts the attempt as a
    boundary rejection (which it physically is - the trajectory
    crossed into a pathological region of phase space). This keeps
    the rest of the chain alive without compromising detailed
    balance (rejection + state restore is always a valid MH move).
    """


WATER_LAMBDA_PARAM = "lambda_water_swap"
"""Name of the global :class:`~openmm.NonbondedForce` parameter that
drives the simultaneous ghost1-decouple / ghost2-couple perturbation.
``0.0`` (the default between attempts) puts ``ghost1`` fully on and
``ghost2`` fully off; ``1.0`` swaps the two. Held at ``0.0`` outside
of an attempt so the production Hamiltonian is always the same."""

SPHERE_RESTRAINT_K_PARAM = "k_water_swap_sphere"
SPHERE_RESTRAINT_R_PARAM = "R_water_swap_sphere"
"""Global parameters for the flat-bottom spherical restraints that keep
each ghost on the correct side of the active-site boundary during the
NCMC switch. ``k`` defaults to ``0`` between attempts; ``R`` tracks the
(currently scaled) active-site radius."""


@dataclass
class WaterSwapSettings:
    """User-tunable knobs for :class:`WaterSwapMC`.

    Defaults follow the production-phase recipe in the Ben-Shalom water-
    swap paper: 0.9 nm active-site sphere; the NCMC switch is run as a
    single 12.8 ps perturbation/propagation interleave; total
    perturbation count and per-perturbation MD steps are tunable so
    the user can shorten the switch for fast tests.

    Parameters
    ----------
    active_site_radius : float
        Radius of the active-site sphere in nanometres. Centred on the
        centre of mass of ``ligand_atom_indices``.
    radius_scales_with_box : bool
        If True (the paper default), ``active_site_radius`` is treated
        as the radius at the construction-time box size and is rescaled
        linearly with the periodic-box ``a`` vector each attempt. With
        a constant volume / NVT simulation this is a no-op.
    n_perturbation_steps : int
        Number of alchemical lambda jumps across the full switch.
        The published recipe is 80 jumps over 12.8 ps; you can dial
        this down for short integration tests.
    n_propagation_steps_per_perturbation : int
        MD steps run between consecutive lambda jumps. At the user's
        4 fs HMR timestep, ``80 * 40 * 4 fs`` = 12.8 ps, matching the
        production switch length.
    direction_probabilities : tuple[float, float]
        Probability of attempting an ``(in, out)`` move. The two values
        must sum to ``1.0``. ``(0.5, 0.5)`` keeps the chain symmetric.
    boundary_check : bool
        If True, reject any move whose alchemically transformed waters
        ended up on the wrong side of the active-site boundary
        (``ghost2`` must finish inside the target region;
        ``ghost1`` must finish inside the source region). The
        published algorithm also rejects mid-switch crossings; the
        cheap end-of-switch version is implemented here.
    pocket_sphere_restraint : bool
        If True, apply harmonic flat-bottomed spherical restraints to
        both ghosts during the NCMC switch so each stays on the correct
        side of the active-site boundary (``ghost2`` inside the pocket
        on insert, outside on delete; ``ghost1`` the reverse). The
        restraints are switched off between attempts so the production
        Hamiltonian is unchanged.
    pocket_sphere_restraint_k : float
        Spring constant for the flat-bottom sphere restraint in
        kJ/mol/nm². Only active while ``pocket_sphere_restraint`` is
        enabled and an NCMC switch is running.
    """

    active_site_radius: float = 0.9
    radius_scales_with_box: bool = True
    n_perturbation_steps: int = 80
    n_propagation_steps_per_perturbation: int = 40
    direction_probabilities: tuple[float, float] = (0.5, 0.5)
    boundary_check: bool = True
    pocket_sphere_restraint: bool = True
    pocket_sphere_restraint_k: float = 4000.0

    def __post_init__(self) -> None:
        if abs(sum(self.direction_probabilities) - 1.0) > 1e-9:
            raise ValueError(
                "direction_probabilities must sum to 1.0, got "
                f"{self.direction_probabilities}"
            )
        if self.n_perturbation_steps < 1:
            raise ValueError("n_perturbation_steps must be >= 1")
        if self.n_propagation_steps_per_perturbation < 0:
            raise ValueError(
                "n_propagation_steps_per_perturbation must be >= 0"
            )


@dataclass
class WaterSwapStats:
    """Per-direction acceptance + boundary-rejection counters.

    Populated by :meth:`WaterSwapMC.attempt`. The mover keeps one of
    these per direction, reachable via :attr:`WaterSwapMC.in_stats` /
    :attr:`WaterSwapMC.out_stats`, plus an aggregated summary at
    :meth:`WaterSwapMC.summary`.
    """

    attempts: int = 0
    accepted: int = 0
    boundary_rejections: int = 0
    empty_source_rejections: int = 0
    work_history: list[float] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of attempts (including instant rejections) that were accepted."""
        if self.attempts == 0:
            return 0.0
        return self.accepted / self.attempts


def _water_oxygen_indices(
    topology: "Topology", water_residue_name: str = "HOH",
) -> list[int]:
    """Return the oxygen atom index of every water residue.

    Used to count waters per region from a single position per molecule
    (the oxygen) rather than from all three atoms; the choice matches
    what most water-swap implementations report and keeps the count
    insensitive to thermal flapping of the H atoms across the boundary.
    """
    indices: list[int] = []
    for residue in topology.residues():
        if residue.name != water_residue_name:
            continue
        for atom in residue.atoms():
            if atom.element is not None and atom.element.symbol == "O":
                indices.append(atom.index)
                break
        else:
            raise ValueError(
                f"Water residue {residue.index} has no oxygen atom; "
                f"atoms: {[a.name for a in residue.atoms()]}"
            )
    return indices


def _water_residue_atoms(
    topology: "Topology", water_residue_name: str = "HOH",
) -> list[list[int]]:
    """Return per-water-residue lists of (O, H, H...) atom indices.

    The ordering inside each sublist follows the topology's atom order
    so that ``_water_residue_atoms(top)[i][0]`` is always the oxygen of
    the i-th water in topology order (matching the order returned by
    :func:`_water_oxygen_indices`).
    """
    out: list[list[int]] = []
    for residue in topology.residues():
        if residue.name != water_residue_name:
            continue
        out.append([a.index for a in residue.atoms()])
    return out


def _periodic_displacement(
    p: np.ndarray, q: np.ndarray, box_lengths: np.ndarray,
) -> np.ndarray:
    """Minimum-image displacement ``p - q`` for an orthorhombic box.

    The water-swap target sites are sampled relative to the ligand COM,
    which is itself just a centroid in the unwrapped coordinate frame,
    so we need the minimum-image convention to compute "how far is this
    water from the ligand" correctly when the simulation cell has been
    crossed since the last image-recentre.
    """
    d = p - q
    d -= box_lengths * np.round(d / box_lengths)
    return d


class WaterSwapMC:
    """NCMC water-swap Monte Carlo mover (two-ghost variant).

    Wraps an existing :class:`openmm.app.Simulation`, a fixed set of
    ligand atoms (defining the active-site centre) and two designated
    "ghost" water residues that bookkeep the swap. ``ghost1`` is the
    "on" slot (always interacting between moves); ``ghost2`` is the
    "off" slot (zero charge / :math:`\\epsilon` between moves). The
    rest of the waters are physical solvent and are never touched by
    the alchemy itself.

    By default the last two water residues in topology order are used
    as ``ghost1`` and ``ghost2`` - that biases the ghost slots away
    from crystallographic waters delivered with the receptor PDB (those
    tend to come first in topology order, are very specifically
    placed, and often play a structural role in the binding pocket).
    The trailing waters are bulk-solvent picks added by Modeller, so
    swapping their identities is safe. Override via
    ``ghost_residue_indices`` when neither end of the topology suits.
    """

    def __init__(
        self,
        simulation: "Simulation",
        ligand_atom_indices: Sequence[int],
        *,
        water_residue_name: str = "HOH",
        ghost_residue_indices: tuple[int, int] | None = None,
        config: WaterSwapSettings | None = None,
        mirror_contexts: Sequence | None = None,
    ) -> None:
        if not ligand_atom_indices:
            raise ValueError(
                "ligand_atom_indices is empty - cannot define active-site centre"
            )
        self.simulation = simulation
        self.config = config or WaterSwapSettings()
        self.water_residue_name = water_residue_name
        self.ligand_atom_indices = np.asarray(list(ligand_atom_indices), dtype=int)
        self.ligand_masses = self._fetch_masses(self.ligand_atom_indices)

        topology = simulation.topology
        self.water_oxygens = np.asarray(
            _water_oxygen_indices(topology, water_residue_name), dtype=int,
        )
        if self.water_oxygens.size < 2:
            raise ValueError(
                "WaterSwapMC needs at least 2 water residues to designate "
                f"as ghost slots; found {self.water_oxygens.size}"
            )
        self.water_atom_groups = _water_residue_atoms(topology, water_residue_name)
        atoms_per_water = {len(g) for g in self.water_atom_groups}
        if atoms_per_water != {3}:
            raise NotImplementedError(
                "WaterSwapMC currently only supports 3-site waters "
                f"(e.g. TIP3P / TIP3P-FB); found waters with atom counts "
                f"{atoms_per_water}"
            )

        if ghost_residue_indices is None:
            # Default to the last two waters in topology order; the
            # first few HOH residues are typically crystallographic
            # waters that ship with the receptor PDB and often play
            # structural roles, so we leave them alone.
            n_waters = self.water_oxygens.size
            ghost_residue_indices = (n_waters - 2, n_waters - 1)
        g1, g2 = ghost_residue_indices
        if g1 == g2:
            raise ValueError("ghost_residue_indices must be two distinct waters")
        if not (
            0 <= g1 < self.water_oxygens.size
            and 0 <= g2 < self.water_oxygens.size
        ):
            raise ValueError(
                f"ghost_residue_indices {ghost_residue_indices!r} out of range "
                f"[0, {self.water_oxygens.size})"
            )
        self.ghost1_water_idx = int(g1)
        self.ghost2_water_idx = int(g2)
        self.ghost1_atoms = np.asarray(
            self.water_atom_groups[self.ghost1_water_idx], dtype=int,
        )
        self.ghost2_atoms = np.asarray(
            self.water_atom_groups[self.ghost2_water_idx], dtype=int,
        )

        self.water_masses = np.asarray(
            [
                [self._particle_mass(idx) for idx in group]
                for group in self.water_atom_groups
            ],
            dtype=float,
        )

        self.nonbonded_force, self.nonbonded_force_index = (
            self._find_nonbonded_force()
        )
        self._setup_alchemy()
        self._setup_sphere_restraints()

        # Mirror the ghost alchemy onto any additional contexts (most
        # importantly the ConstantPH ``relaxationContext``, whose
        # system is a deep-copy of the explicit system taken *before*
        # WaterSwapMC was constructed). Without this mirror, ghost2 -
        # which has no forces in the production simulation and so
        # random-walks freely between attempts - would still carry
        # full TIP3P parameters in the relaxation system, and the
        # first relaxation MD step after ghost2 drifts onto another
        # heavy atom NaN-bombs. Mirroring keeps ghost2 decoupled
        # everywhere at ``lambda_water_swap = 0`` (the inter-attempt
        # value) so the relaxation context sees the same effective
        # Hamiltonian as the production context.
        self._mirror_contexts: list = []
        if mirror_contexts is not None:
            for mirror_ctx in mirror_contexts:
                self._install_alchemy_on_context(mirror_ctx)
                self._mirror_contexts.append(mirror_ctx)

        # Pre-compute heavy-atom indices for clash rejection: any
        # non-water atom (protein, ligand, ions) and the oxygen of
        # every water. Hydrogens are omitted because the water-water
        # rejection threshold (~0.28 nm) is anchored on the O LJ
        # well and hydrogens always sit within constraint distance
        # of an oxygen.
        water_atom_idx_set = {
            idx for group in self.water_atom_groups for idx in group
        }
        non_water_heavy = []
        for atom in topology.atoms():
            if atom.index in water_atom_idx_set:
                continue
            element = atom.element
            if element is None:
                continue
            if element.symbol == "H":
                continue
            non_water_heavy.append(atom.index)
        self._clash_heavy_atom_indices = np.asarray(non_water_heavy, dtype=int)

        self.integrator = simulation.integrator
        self.temperature = self.integrator.getTemperature()
        self.kT = MOLAR_GAS_CONSTANT_R * self.temperature

        box_vectors = (
            simulation.context.getState(getPositions=False).getPeriodicBoxVectors()
        )
        self._reference_box_a = float(box_vectors[0].x)

        self.in_stats = WaterSwapStats()
        self.out_stats = WaterSwapStats()

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    @property
    def total_attempts(self) -> int:
        return self.in_stats.attempts + self.out_stats.attempts

    @property
    def total_accepted(self) -> int:
        return self.in_stats.accepted + self.out_stats.accepted

    @property
    def acceptance_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.total_accepted / self.total_attempts

    def reset_stats(self) -> None:
        """Zero acceptance counters and clear the work history."""
        self.in_stats = WaterSwapStats()
        self.out_stats = WaterSwapStats()

    def summary(self) -> dict[str, float]:
        """Return a flat summary dict suitable for logging."""
        return {
            "total_attempts": float(self.total_attempts),
            "total_accepted": float(self.total_accepted),
            "acceptance_rate": float(self.acceptance_rate),
            "in_attempts": float(self.in_stats.attempts),
            "in_accepted": float(self.in_stats.accepted),
            "in_acceptance_rate": float(self.in_stats.acceptance_rate),
            "in_boundary_rejections": float(self.in_stats.boundary_rejections),
            "in_empty_source_rejections": float(
                self.in_stats.empty_source_rejections
            ),
            "out_attempts": float(self.out_stats.attempts),
            "out_accepted": float(self.out_stats.accepted),
            "out_acceptance_rate": float(self.out_stats.acceptance_rate),
            "out_boundary_rejections": float(self.out_stats.boundary_rejections),
            "out_empty_source_rejections": float(
                self.out_stats.empty_source_rejections
            ),
        }

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _find_nonbonded_force(self) -> tuple[NonbondedForce, int]:
        """Locate the standard :class:`NonbondedForce`.

        The alchemy is rerouted through this force's particle-parameter
        offsets, which work with PME and have a low overhead compared
        to adding an extra :class:`CustomNonbondedForce`. ConstantPH's
        REST2 hookup installs additional offsets on this same force
        (for the *ligand* atoms only) - those are kept untouched here.
        """
        system = self.simulation.system
        candidates = [
            (i, f)
            for i, f in enumerate(system.getForces())
            if isinstance(f, NonbondedForce)
        ]
        if not candidates:
            raise RuntimeError(
                "WaterSwapMC requires a NonbondedForce in the system"
            )
        if len(candidates) > 1:
            raise RuntimeError(
                "WaterSwapMC found multiple NonbondedForce instances; "
                "an unambiguous force is required"
            )
        return candidates[0][1], candidates[0][0]

    def _setup_alchemy(self) -> None:
        """Install the persistent alchemical offsets for the two ghosts.

        Modifies the explicit-system :class:`NonbondedForce` in place:

        * Adds ``lambda_water_swap`` as a global parameter, default ``0``.
        * For each ``ghost1`` atom: adds an offset with chargeScale
          ``-q`` and epsilonScale ``-eps``. At ``lambda=0`` the offset
          contributes nothing (ghost1 stays fully coupled); at
          ``lambda=1`` it cancels the base parameters so ghost1 goes
          fully ghost-like.
        * For each ``ghost2`` atom: zeros out the base ``charge`` and
          ``epsilon``, then adds an offset with chargeScale ``+q_orig``
          and epsilonScale ``+eps_orig``. At ``lambda=0`` ghost2 is
          fully decoupled; at ``lambda=1`` it has its original
          parameters restored. ``sigma`` is left untouched throughout
          so the rigid-water constraint and intramolecular geometry
          are unaffected.

        The context is :meth:`reinitialize`d (with
        ``preserveState=True``) so the new global parameter is visible
        to subsequent :meth:`Context.setParameter` calls. Positions,
        velocities and box vectors are preserved across the
        reinitialise.
        """
        force = self.nonbonded_force
        existing_globals = {
            force.getGlobalParameterName(i)
            for i in range(force.getNumGlobalParameters())
        }
        if WATER_LAMBDA_PARAM not in existing_globals:
            force.addGlobalParameter(WATER_LAMBDA_PARAM, 0.0)

        ghost1_offset_indices: list[int] = []
        for atom_idx in self.ghost1_atoms:
            charge, _sigma, epsilon = force.getParticleParameters(int(atom_idx))
            q = float(charge.value_in_unit(unit.elementary_charge))
            eps = float(epsilon.value_in_unit(unit.kilojoule_per_mole))
            ghost1_offset_indices.append(
                force.addParticleParameterOffset(
                    WATER_LAMBDA_PARAM,
                    int(atom_idx),
                    -q, 0.0, -eps,
                )
            )

        ghost2_offset_indices: list[int] = []
        for atom_idx in self.ghost2_atoms:
            charge, sigma, epsilon = force.getParticleParameters(int(atom_idx))
            q_orig = float(charge.value_in_unit(unit.elementary_charge))
            eps_orig = float(epsilon.value_in_unit(unit.kilojoule_per_mole))
            force.setParticleParameters(int(atom_idx), 0.0, sigma, 0.0)
            ghost2_offset_indices.append(
                force.addParticleParameterOffset(
                    WATER_LAMBDA_PARAM,
                    int(atom_idx),
                    q_orig, 0.0, eps_orig,
                )
            )

        self._ghost1_offset_indices = ghost1_offset_indices
        self._ghost2_offset_indices = ghost2_offset_indices

        self.simulation.context.reinitialize(preserveState=True)
        self.simulation.context.setParameter(WATER_LAMBDA_PARAM, 0.0)

    def _setup_sphere_restraints(self) -> None:
        """Install flat-bottom spherical restraints on both ghost oxygens.

        Each bond ties a ghost oxygen centroid to the ligand COM centroid.
        A per-bond ``mode`` parameter selects the flat-bottom direction:
        ``+1`` penalises ``r > R`` (keep inside the pocket sphere),
        ``-1`` penalises ``r < R`` (keep outside). The global spring
        constant ``k`` is held at ``0`` between attempts so the
        production Hamiltonian is unchanged.
        """
        self._sphere_restraint_force_index: int | None = None
        self._sphere_restraint_ghost2_bond = 0
        self._sphere_restraint_ghost1_bond = 1
        if not self.config.pocket_sphere_restraint:
            return

        expr = (
            f"0.5*{SPHERE_RESTRAINT_K_PARAM}*("
            f"step(mode)*max(0, distance(g1,g2)-{SPHERE_RESTRAINT_R_PARAM})^2 + "
            f"step(-mode)*max(0, {SPHERE_RESTRAINT_R_PARAM}-distance(g1,g2))^2)"
        )
        force = CustomCentroidBondForce(2, expr)
        force.addGlobalParameter(SPHERE_RESTRAINT_K_PARAM, 0.0)
        force.addGlobalParameter(
            SPHERE_RESTRAINT_R_PARAM, float(self.config.active_site_radius),
        )
        force.addPerBondParameter("mode")
        force.setUsesPeriodicBoundaryConditions(True)

        ligand_indices = [int(i) for i in self.ligand_atom_indices]
        ligand_weights = [float(m) for m in self.ligand_masses]
        ligand_group = force.addGroup(ligand_indices, ligand_weights)

        ghost2_o = int(self.water_oxygens[self.ghost2_water_idx])
        ghost1_o = int(self.water_oxygens[self.ghost1_water_idx])
        ghost2_group = force.addGroup([ghost2_o])
        ghost1_group = force.addGroup([ghost1_o])

        force.addBond([ghost2_group, ligand_group], [1.0])
        force.addBond([ghost1_group, ligand_group], [-1.0])

        self._sphere_restraint_force_index = self.simulation.system.addForce(force)
        self.simulation.context.reinitialize(preserveState=True)
        ctx = self.simulation.context
        ctx.setParameter(SPHERE_RESTRAINT_K_PARAM, 0.0)
        ctx.setParameter(
            SPHERE_RESTRAINT_R_PARAM, float(self.config.active_site_radius),
        )

    def _sphere_restraint_force(self) -> CustomCentroidBondForce:
        if self._sphere_restraint_force_index is None:
            raise RuntimeError("sphere restraints are not installed")
        force = self.simulation.system.getForce(self._sphere_restraint_force_index)
        if not isinstance(force, CustomCentroidBondForce):
            raise RuntimeError(
                "expected CustomCentroidBondForce at "
                f"index {self._sphere_restraint_force_index}"
            )
        return force

    def _activate_sphere_restraints(
        self, *, direction_is_in: bool, radius: float,
    ) -> None:
        """Turn on flat-bottom sphere restraints for an NCMC switch."""
        if self._sphere_restraint_force_index is None:
            return
        cfg = self.config
        ctx = self.simulation.context
        ctx.setParameter(SPHERE_RESTRAINT_K_PARAM, float(cfg.pocket_sphere_restraint_k))
        ctx.setParameter(SPHERE_RESTRAINT_R_PARAM, float(radius))

        force = self._sphere_restraint_force()
        if direction_is_in:
            g2_groups, _ = force.getBondParameters(self._sphere_restraint_ghost2_bond)
            g1_groups, _ = force.getBondParameters(self._sphere_restraint_ghost1_bond)
            force.setBondParameters(self._sphere_restraint_ghost2_bond, g2_groups, [1.0])
            force.setBondParameters(self._sphere_restraint_ghost1_bond, g1_groups, [-1.0])
        else:
            g2_groups, _ = force.getBondParameters(self._sphere_restraint_ghost2_bond)
            g1_groups, _ = force.getBondParameters(self._sphere_restraint_ghost1_bond)
            force.setBondParameters(self._sphere_restraint_ghost2_bond, g2_groups, [-1.0])
            force.setBondParameters(self._sphere_restraint_ghost1_bond, g1_groups, [1.0])
        force.updateParametersInContext(ctx)

    def _deactivate_sphere_restraints(self) -> None:
        """Switch off sphere restraints (``k = 0``)."""
        if self._sphere_restraint_force_index is None:
            return
        self.simulation.context.setParameter(SPHERE_RESTRAINT_K_PARAM, 0.0)

    def _install_alchemy_on_context(self, context) -> None:
        """Apply the same two-ghost alchemy to an additional context.

        Intended for sister contexts that share the production
        topology atom ordering but were built from a separately
        constructed ``System`` (typically ConstantPH's
        ``relaxationContext``, whose system is a ``deepcopy`` of the
        explicit system taken *before* WaterSwapMC was instantiated).
        We do an in-place edit of that context's :class:`NonbondedForce`:
        add ``lambda_water_swap`` as a global parameter, install the
        ghost1 / ghost2 particle-parameter offsets, zero out ghost2's
        base nonbonded params, then :meth:`reinitialize` the context
        with ``preserveState=True`` so the new global parameter
        becomes visible. The mirror context is finally pinned to
        ``lambda_water_swap = 0`` to match the production context's
        between-attempts value (it stays at zero forever - WaterSwapMC
        never drives lambda on a mirror context).
        """
        system = context.getSystem()
        candidates = [
            f for f in system.getForces()
            if isinstance(f, NonbondedForce)
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "mirror_contexts entry must wrap a system with exactly "
                f"one NonbondedForce; got {len(candidates)}"
            )
        force = candidates[0]
        existing_globals = {
            force.getGlobalParameterName(i)
            for i in range(force.getNumGlobalParameters())
        }
        if WATER_LAMBDA_PARAM not in existing_globals:
            force.addGlobalParameter(WATER_LAMBDA_PARAM, 0.0)
        for atom_idx in self.ghost1_atoms:
            charge, _sigma, epsilon = force.getParticleParameters(int(atom_idx))
            q = float(charge.value_in_unit(unit.elementary_charge))
            eps = float(epsilon.value_in_unit(unit.kilojoule_per_mole))
            force.addParticleParameterOffset(
                WATER_LAMBDA_PARAM,
                int(atom_idx),
                -q, 0.0, -eps,
            )
        for atom_idx in self.ghost2_atoms:
            charge, sigma, epsilon = force.getParticleParameters(int(atom_idx))
            q_orig = float(charge.value_in_unit(unit.elementary_charge))
            eps_orig = float(epsilon.value_in_unit(unit.kilojoule_per_mole))
            force.setParticleParameters(int(atom_idx), 0.0, sigma, 0.0)
            force.addParticleParameterOffset(
                WATER_LAMBDA_PARAM,
                int(atom_idx),
                q_orig, 0.0, eps_orig,
            )
        context.reinitialize(preserveState=True)
        context.setParameter(WATER_LAMBDA_PARAM, 0.0)

    def _fetch_masses(self, atom_indices: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self._particle_mass(i) for i in atom_indices], dtype=float,
        )

    def _particle_mass(self, atom_index: int) -> float:
        m = self.simulation.system.getParticleMass(int(atom_index))
        return float(m.value_in_unit(unit.dalton))

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _box_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Read positions+velocities+box and derive ``V_box`` and the radius.

        Returns
        -------
        positions_nm : np.ndarray, shape (N, 3)
        velocities_nm_per_ps : np.ndarray, shape (N, 3)
        box_lengths_nm : np.ndarray, shape (3,)
        v_box_nm3 : float
        radius_nm : float
        """
        state = self.simulation.context.getState(
            getPositions=True,
            getVelocities=True,
            enforcePeriodicBox=False,
        )
        positions = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=float,
        )
        velocities = np.asarray(
            state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond,
            ),
            dtype=float,
        )
        box_vectors = np.asarray(
            [
                v.value_in_unit(unit.nanometer)
                for v in state.getPeriodicBoxVectors()
            ],
            dtype=float,
        )
        box_lengths = np.array(
            [box_vectors[0][0], box_vectors[1][1], box_vectors[2][2]], dtype=float,
        )
        v_box = float(abs(np.linalg.det(box_vectors)))
        if self.config.radius_scales_with_box and self._reference_box_a > 0.0:
            scale = box_lengths[0] / self._reference_box_a
        else:
            scale = 1.0
        radius = float(self.config.active_site_radius * scale)
        return positions, velocities, box_lengths, v_box, radius

    def _ligand_center(self, positions: np.ndarray) -> np.ndarray:
        ligand_positions = positions[self.ligand_atom_indices]
        total_mass = float(self.ligand_masses.sum())
        if total_mass <= 0.0:
            return ligand_positions.mean(axis=0)
        return (ligand_positions * self.ligand_masses[:, None]).sum(axis=0) / total_mass

    def _active_site_mask(
        self,
        positions: np.ndarray,
        box_lengths: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """Boolean mask of water residues whose O atom is inside the sphere.
        Does NOT use PBC"""
        oxygens = positions[self.water_oxygens]
        diffs = oxygens - center
        diffs -= box_lengths * np.round(diffs / box_lengths)
        return (diffs ** 2).sum(axis=1) <= radius ** 2

    def _count_on_waters(
        self,
        mask_active: np.ndarray,
        off_ghost_water_idx: int,
    ) -> tuple[int, int]:
        """Return ``(n_active, n_bulk)`` for *interacting* waters.

        The off ghost is excluded from both counts because it carries
        no charge / epsilon and so does not interact with the system.
        Use ``off_ghost_water_idx = self.ghost2_water_idx`` to count
        at ``t0`` (where ghost2 is the off slot) and
        ``off_ghost_water_idx = self.ghost1_water_idx`` to count at
        ``tEnd`` (where ghost1 has been decoupled).
        """
        mask = mask_active.copy()
        if 0 <= off_ghost_water_idx < mask.size:
            mask[off_ghost_water_idx] = False
            n_active = int(mask.sum())
            n_total_on = int(self.water_oxygens.size - 1)
            n_bulk = n_total_on - n_active
        else:
            n_active = int(mask.sum())
            n_bulk = int(self.water_oxygens.size - n_active)
        return n_active, n_bulk

    MIN_OXYGEN_GAP_NM = 0.295
    """Minimum allowed distance between the proposed ghost2 oxygen and any
    real solvent oxygen, in nanometres. Set slightly inside the
    TIP3P O-O LJ minimum (~0.318 nm); placing ghost2 closer than this
    on top of a real water produces an LJ force spike that NaN-bombs
    the integrator without soft-core, and even short of NaN the
    accumulated linear-coupling work runs to tens of kT per attempt
    so the Metropolis test consistently rejects. Using ~0.28 nm
    keeps the freshly-placed water near (rather than inside) the LJ
    well so the work distribution stays mostly within +/- 10 kT.
    The same threshold is used for both ``in`` and ``out`` target
    sampling, so any detailed-balance bias from the rejection
    sampling cancels between forward and reverse moves.
    """

    MAX_TARGET_RESAMPLE_ATTEMPTS = 1000

    def _propose_active_site_point(
        self,
        center: np.ndarray,
        radius: float,
        positions: np.ndarray,
        box_lengths: np.ndarray,
    ) -> np.ndarray | None:
        """Sample a uniform-but-clash-free point inside the active-site sphere.

        Uses the radius-cubed CDF trick so the underlying density is
        uniform per unit volume (without it the points would cluster
        near the centre and bias the Jacobian), then rejects any
        proposal that lands within
        :attr:`MIN_OXYGEN_GAP_NM` of an existing solvent oxygen.

        Returns ``None`` if no clash-free point was found in
        :attr:`MAX_TARGET_RESAMPLE_ATTEMPTS` tries - the caller
        treats that as a soft rejection rather than raising.
        """
        rng = np.random.random
        oxygens = positions[self.water_oxygens]
        for _ in range(self.MAX_TARGET_RESAMPLE_ATTEMPTS):
            u = rng()
            r = radius * u ** (1.0 / 3.0)
            cos_theta = 2.0 * rng() - 1.0
            sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta ** 2)))
            phi = 2.0 * np.pi * rng()
            offset = np.array(
                [
                    r * sin_theta * np.cos(phi),
                    r * sin_theta * np.sin(phi),
                    r * cos_theta,
                ],
                dtype=float,
            )
            candidate = center + offset
            if self._is_clear_of_oxygens(
                candidate, oxygens, box_lengths, positions,
            ):
                return candidate
        return None

    def _propose_bulk_point(
        self,
        center: np.ndarray,
        radius: float,
        box_lengths: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray | None:
        """Sample a uniform-but-clash-free point in the box, outside the sphere.

        Uses rejection sampling on a uniform-in-box draw; the box is
        much larger than the active-site sphere in practice so the
        acceptance rate of "outside the sphere" is essentially one,
        and the dominant rejection cause is being too close to an
        existing solvent oxygen. Returns ``None`` if no clash-free
        bulk point was found in
        :attr:`MAX_TARGET_RESAMPLE_ATTEMPTS` tries.
        """
        rng = np.random.random
        oxygens = positions[self.water_oxygens]
        for _ in range(self.MAX_TARGET_RESAMPLE_ATTEMPTS):
            point = np.array(
                [
                    (rng() - 0.5) * box_lengths[0],
                    (rng() - 0.5) * box_lengths[1],
                    (rng() - 0.5) * box_lengths[2],
                ],
                dtype=float,
            ) + center
            d = _periodic_displacement(point, center, box_lengths)
            if float((d ** 2).sum()) <= radius ** 2:
                continue
            if self._is_clear_of_oxygens(
                point, oxygens, box_lengths, positions,
            ):
                return point
        return None

    def _is_clear_of_oxygens(
        self,
        point: np.ndarray,
        oxygens: np.ndarray,
        box_lengths: np.ndarray,
        positions: np.ndarray | None = None,
    ) -> bool:
        """Return True iff ``point`` is at least ``MIN_OXYGEN_GAP_NM`` from any
        solvent oxygen *and* any non-water heavy atom under
        minimum-image distances.

        The same threshold is used for water O's and protein/ligand
        heavy atoms. That is a coarse approximation - protein heavy
        atoms have a range of LJ sigma values - but works in
        practice because the binding-pocket atoms are mostly C / N /
        O / S whose LJ sigmas all sit within +/- 20% of water's
        0.315 nm. Without this protein-side check the eval scripts
        regularly NaN out when ghost2 is dropped on top of an
        unfortunate sidechain.

        ``positions`` is the full (N, 3) positions array, used to
        index :attr:`_clash_heavy_atom_indices`. It can be ``None``
        for the unit-test setup where the system contains only
        waters; the method then reduces to the water-only check.
        """
        threshold = self.MIN_OXYGEN_GAP_NM ** 2
        diffs = oxygens - point
        diffs -= box_lengths * np.round(diffs / box_lengths)
        d2 = (diffs ** 2).sum(axis=1)
        # Exclude near-zero distances (i.e. the point coincides with
        # the ghost's own current oxygen position) by treating any
        # distance smaller than 1e-16 nm^2 as the ghost's own atom.
        if not bool(np.all((d2 < 1e-16) | (d2 >= threshold))):
            return False
        if (
            positions is not None
            and self._clash_heavy_atom_indices.size > 0
        ):
            heavies = positions[self._clash_heavy_atom_indices]
            diffs_h = heavies - point
            diffs_h -= box_lengths * np.round(diffs_h / box_lengths)
            d2_h = (diffs_h ** 2).sum(axis=1)
            if not bool(np.all(d2_h >= threshold)):
                return False
        return True

    # ------------------------------------------------------------------
    # Position / state mutators (in-place on a positions/velocities array)
    # ------------------------------------------------------------------

    def _translate_water(
        self,
        positions: np.ndarray,
        water_residue_idx: int,
        target_o_position: np.ndarray,
    ) -> None:
        """Shift all 3 atoms of a water by ``(target - current_O)``.

        Operates on a positions array in-place; preserves the
        intramolecular rigid-water geometry exactly (because the same
        translation vector is applied to all three atoms). Velocities
        are intentionally not touched - momentum is preserved across
        the translation.
        """
        atoms = self.water_atom_groups[water_residue_idx]
        o_idx = atoms[0]
        delta = target_o_position - positions[o_idx]
        for idx in atoms:
            positions[idx] = positions[idx] + delta

    def _swap_waters(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        water_a: int,
        water_b: int,
    ) -> None:
        """Swap positions and velocities of waters ``a`` and ``b``.

        Equivalent to a label permutation: for two identical
        three-site waters the atomic positions and velocities swap
        wholesale (atom-by-atom). The energy of the system is
        unchanged when both waters have the same nonbonded parameters
        and the same constraint set, so the swap is free; when one is
        fully coupled and the other fully decoupled (the post-NCMC
        case) the swap also preserves the energy because the on-water
        ends up at the same spatial position - only the label of the
        "on" slot moves.
        """
        atoms_a = self.water_atom_groups[water_a]
        atoms_b = self.water_atom_groups[water_b]
        for a_idx, b_idx in zip(atoms_a, atoms_b, strict=True):
            pa = positions[a_idx].copy()
            positions[a_idx] = positions[b_idx]
            positions[b_idx] = pa
            va = velocities[a_idx].copy()
            velocities[a_idx] = velocities[b_idx]
            velocities[b_idx] = va

    # ------------------------------------------------------------------
    # NCMC switching
    # ------------------------------------------------------------------

    def _run_switch(self) -> float:
        """Drive ``lambda_water_swap`` from ``0 -> 1`` and accumulate the work.

        At each of ``n_perturbation_steps`` jumps:

        1. Snapshot potential energy at the current lambda.
        2. Step lambda to the next scheduled value.
        3. Snapshot potential energy at the new lambda.
        4. Add ``(E_after - E_before) / kT`` to the work.
        5. Run ``n_propagation_steps_per_perturbation`` MD steps (which
           do not contribute to the work because they happen at fixed
           parameters; they merely relax the system between jumps).

        Raises :class:`WaterSwapSwitchFailed` if the switch produces
        a NaN energy or NaN positions (typically when the linear
        alchemy without soft-core LJ ramps ghost2 onto a heavy atom
        and the constraint solver can't absorb the force); the
        caller treats that as a hard rejection and rolls the context
        back to the snapshot.
        """
        ctx = self.simulation.context
        cfg = self.config
        n_pert = cfg.n_perturbation_steps
        n_prop = cfg.n_propagation_steps_per_perturbation

        kT_value = self.kT.value_in_unit(unit.kilojoule_per_mole)
        ctx.setParameter(WATER_LAMBDA_PARAM, 0.0)
        work = 0.0
        for k in range(n_pert):
            lam_new = (k + 1) / n_pert
            energy_before = self._potential_energy()
            if not np.isfinite(energy_before):
                raise WaterSwapSwitchFailed(
                    f"non-finite energy at step {k} before lambda jump"
                )
            ctx.setParameter(WATER_LAMBDA_PARAM, float(lam_new))
            energy_after = self._potential_energy()
            if not np.isfinite(energy_after):
                raise WaterSwapSwitchFailed(
                    f"non-finite energy at step {k} after lambda jump"
                )
            work += (energy_after - energy_before) / kT_value
            if n_prop > 0:
                try:
                    self.simulation.step(n_prop)
                except Exception as exc:  # noqa: BLE001
                    raise WaterSwapSwitchFailed(
                        f"integrator step failed at NCMC step {k}: {exc}"
                    ) from exc
        return work

    def _potential_energy(self) -> float:
        return float(
            self.simulation.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def attempt(self) -> bool:
        """Try one NCMC water-swap move; return True iff accepted.

        Workflow (per the module docstring):

        1. Snapshot positions / velocities / box / lambda.
        2. Pick a direction (``in`` or ``out``) and tally
           ``N_act(t0)``, ``N_B(t0)`` excluding the currently-off
           ``ghost2``.
        3. Stage the move:

           * ``in``: translate ``ghost2`` to a uniform-random
             active-site point.
           * ``out``: swap ``ghost1`` with a random *real* pocket
             water, then translate ``ghost2`` to a uniform-random
             bulk point.

           Write the staged positions + velocities back to the
           context so the NCMC switch starts from them.
        4. Run the simultaneous ``lambda_water_swap = 0 -> 1`` switch,
           accumulating the dimensionless work.
        5. Tally ``N_act(tEnd)``, ``N_B(tEnd)`` excluding the now-off
           ``ghost1``; run the boundary check (``ghost2`` must be in
           the target region, ``ghost1`` must be in the source
           region).
        6. Apply the Metropolis-Hastings test with the paper's volume
           / count Jacobian. On acceptance, swap ghost1/ghost2 atomic
           positions + velocities and reset ``lambda_water_swap`` to
           ``0`` so the invariant ``ghost1 = on``, ``ghost2 = off`` is
           restored with ``ghost1`` now sitting in the target region.
        """
        cfg = self.config
        ctx = self.simulation.context
        rng = np.random.random

        positions, velocities, box_lengths, v_box, radius = self._box_state()
        center = self._ligand_center(positions)
        mask_active = self._active_site_mask(
            positions, box_lengths, center, radius,
        )

        n_active_t0, n_bulk_t0 = self._count_on_waters(
            mask_active, off_ghost_water_idx=self.ghost2_water_idx,
        )

        v_act = (4.0 / 3.0) * np.pi * radius ** 3
        v_bulk = max(v_box - v_act, 1e-12)

        p_in, _p_out = cfg.direction_probabilities
        direction_is_in = rng() < p_in
        stats = self.in_stats if direction_is_in else self.out_stats
        stats.attempts += 1

        snapshot_state = ctx.getState(
            getPositions=True,
            getVelocities=True,
            enforcePeriodicBox=False,
        )
        snapshot_lambda = ctx.getParameter(WATER_LAMBDA_PARAM)

        # ------------------------------------------------------------
        # Stage the move so the NCMC switch starts from the canonical
        # configuration: ghost1 sits in the source region (on) and
        # ghost2 sits in the target region (off). The free-energy
        # cost of every staging operation is zero - the W <-> ghost1
        # swap is a pure relabelling of two identical on-waters, and
        # translating ghost2 costs nothing because ghost2 carries no
        # charge / epsilon between attempts.
        # ------------------------------------------------------------
        staged_positions = positions.copy()
        staged_velocities = velocities.copy()
        staged_mask_active = mask_active.copy()

        if direction_is_in:
            # ghost1 must be in *bulk* before NCMC. If MD pushed it
            # into the pocket (or it ended up there after a previous
            # accepted in-move), relabel by swapping atoms with a
            # random *real* bulk water; the energy is unchanged and
            # the on-water set is identical.
            if staged_mask_active[self.ghost1_water_idx]:
                bulk_relabel_candidates = [
                    int(i)
                    for i in np.flatnonzero(~staged_mask_active)
                    if i != self.ghost2_water_idx
                ]
                if not bulk_relabel_candidates:
                    stats.empty_source_rejections += 1
                    return False
                chosen_relabel = int(np.random.choice(bulk_relabel_candidates))
                self._swap_waters(
                    staged_positions,
                    staged_velocities,
                    self.ghost1_water_idx,
                    chosen_relabel,
                )
                staged_mask_active[self.ghost1_water_idx] = False
                staged_mask_active[chosen_relabel] = True
            target_point = self._propose_active_site_point(
                center, radius, staged_positions, box_lengths,
            )
            if target_point is None:
                stats.empty_source_rejections += 1
                return False
            self._translate_water(
                staged_positions, self.ghost2_water_idx, target_point,
            )
            staged_mask_active[self.ghost2_water_idx] = True
        else:
            # Out: pick a random *interacting* water in the pocket -
            # ghost1 itself counts as a valid pick (a no-op swap),
            # but ghost2 (off) is excluded because it does not
            # interact. ``W <-> ghost1`` is a free relabelling that
            # leaves ghost1 at a pocket position regardless of where
            # MD had pushed it.
            pocket_pick_candidates = [
                int(i)
                for i in np.flatnonzero(staged_mask_active)
                if i != self.ghost2_water_idx
            ]
            if not pocket_pick_candidates:
                stats.empty_source_rejections += 1
                return False
            chosen_pocket_water = int(np.random.choice(pocket_pick_candidates))
            if chosen_pocket_water != self.ghost1_water_idx:
                ghost1_was_active = bool(staged_mask_active[self.ghost1_water_idx])
                self._swap_waters(
                    staged_positions,
                    staged_velocities,
                    self.ghost1_water_idx,
                    chosen_pocket_water,
                )
                staged_mask_active[self.ghost1_water_idx] = True
                staged_mask_active[chosen_pocket_water] = ghost1_was_active
            target_point = self._propose_bulk_point(
                center, radius, box_lengths, staged_positions,
            )
            if target_point is None:
                stats.empty_source_rejections += 1
                return False
            self._translate_water(
                staged_positions, self.ghost2_water_idx, target_point,
            )
            staged_mask_active[self.ghost2_water_idx] = False

        ctx.setPositions(staged_positions * unit.nanometer)
        ctx.setVelocities(
            staged_velocities * (unit.nanometer / unit.picosecond),
        )
        ctx.setParameter(WATER_LAMBDA_PARAM, 0.0)

        # ------------------------------------------------------------
        # NCMC switch.
        # ------------------------------------------------------------
        self._activate_sphere_restraints(
            direction_is_in=direction_is_in, radius=radius,
        )
        try:
            work = self._run_switch()
        except WaterSwapSwitchFailed:
            stats.boundary_rejections += 1
            self._restore_snapshot(snapshot_state, snapshot_lambda)
            return False
        except Exception:
            self._restore_snapshot(snapshot_state, snapshot_lambda)
            raise
        finally:
            self._deactivate_sphere_restraints()

        # ------------------------------------------------------------
        # End-of-switch tallies + boundary check.
        # ------------------------------------------------------------
        (
            positions_end, velocities_end, box_lengths_end,
            v_box_end, radius_end,
        ) = self._box_state()
        center_end = self._ligand_center(positions_end)
        mask_active_end = self._active_site_mask(
            positions_end, box_lengths_end, center_end, radius_end,
        )

        n_active_end, n_bulk_end = self._count_on_waters(
            mask_active_end, off_ghost_water_idx=self.ghost1_water_idx,
        )

        v_act_end = (4.0 / 3.0) * np.pi * radius_end ** 3
        v_bulk_end = max(v_box_end - v_act_end, 1e-12)

        ghost1_in_active_end = bool(mask_active_end[self.ghost1_water_idx])
        ghost2_in_active_end = bool(mask_active_end[self.ghost2_water_idx])
        if cfg.boundary_check:
            # After staging both ghosts are placed in their canonical
            # regions (ghost1 in source, ghost2 in target); the
            # boundary check enforces that the alchemical switching
            # itself did not push either across the boundary.
            ghost1_started_active = bool(
                staged_mask_active[self.ghost1_water_idx]
            )
            ghost2_started_active = bool(
                staged_mask_active[self.ghost2_water_idx]
            )
            if (
                ghost1_in_active_end != ghost1_started_active
                or ghost2_in_active_end != ghost2_started_active
            ):
                stats.boundary_rejections += 1
                self._restore_snapshot(snapshot_state, snapshot_lambda)
                return False

        if direction_is_in:
            numerator = n_bulk_t0 * v_act
            denominator = n_active_end * v_bulk_end
        else:
            numerator = n_active_t0 * v_bulk
            denominator = n_bulk_end * v_act_end

        if denominator <= 0.0:
            self._restore_snapshot(snapshot_state, snapshot_lambda)
            return False

        log_jacobian = float(np.log(numerator / denominator))
        log_p = log_jacobian - work
        stats.work_history.append(float(work))

        if log_p >= 0.0 or rng() < float(np.exp(log_p)):
            # Accept: relabel ghost1 <-> ghost2 by swapping their
            # atomic positions+velocities, then reset lambda. After
            # this the on-slot (ghost1) is at the target-region
            # position; the off-slot (ghost2) is at the source-region
            # position. This is a pure relabelling - the physical
            # positions of "the on water" and "the off water" do not
            # change - so it adds zero work and no further energy
            # update is required.
            positions_final = positions_end.copy()
            velocities_final = velocities_end.copy()
            self._swap_waters(
                positions_final,
                velocities_final,
                self.ghost1_water_idx,
                self.ghost2_water_idx,
            )
            ctx.setPositions(positions_final * unit.nanometer)
            ctx.setVelocities(
                velocities_final * (unit.nanometer / unit.picosecond),
            )
            ctx.setParameter(WATER_LAMBDA_PARAM, 0.0)
            stats.accepted += 1
            return True

        self._restore_snapshot(snapshot_state, snapshot_lambda)
        return False

    def _restore_snapshot(
        self,
        snapshot_state,
        snapshot_lambda: float,
    ) -> None:
        """Roll back positions / velocities / box / lambda to the snapshot.

        Velocities are restored as well so that a rejected NCMC switch
        does not leak its in-switch momentum into the next MD block;
        the lambda value goes back to whatever it was on entry (almost
        always ``0``) so that the global parameter does not drift even
        after long unsuccessful runs.
        """
        ctx = self.simulation.context
        ctx.setPositions(snapshot_state.getPositions())
        ctx.setVelocities(snapshot_state.getVelocities())
        ctx.setPeriodicBoxVectors(*snapshot_state.getPeriodicBoxVectors())
        ctx.setParameter(WATER_LAMBDA_PARAM, float(snapshot_lambda))
