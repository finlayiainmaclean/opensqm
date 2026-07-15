r"""pH replica-exchange wrapper around :class:`~opensqm.cph.constantph.ConstantPH`.

Each replica carries the full pH ladder and may walk it via simulated
tempering (as in a single-replica :class:`ConstantPH` run).  Adjacent
replicas periodically attempt to exchange their full thermodynamic states
(coordinates, velocities, protonation pattern, and active pH index).

The acceptance probability for exchanging replicas *i* and *j* depends
only on the difference in titratable protons and the pH values, as in
explicit-solvent CpHMD (Mongan *et al.*, eq. 3):

.. math::

    P_\\mathrm{acc} = \\min\\!\\left(1,\\;
        10^{(N_i - N_j)(\\mathrm{pH}_j - \\mathrm{pH}_i)}\\right)

Because the underlying potential at fixed protonation does not depend on
the replica pH label, no explicit-solvent energy difference enters the
exchange criterion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from openmm.unit import nanometers

from opensqm.cph.constantph import ConstantPH

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from openff.toolkit.topology import Molecule  # type: ignore
    from openmm import Platform, State, unit
    from openmm.app import Topology

    from opensqm.cph.reference_energy.models import TitratableResidueReference
    from opensqm.cph.simulation_config import ConstantpHSettings


def num_titratable_protons(cph: ConstantPH) -> int:
    """Count titratable protons present across all driven residues."""
    return sum(t.explicit_states[t.current_index].num_hydrogens for t in cph.titrations.values())


def replica_exchange_log_probability(cph_i: ConstantPH, cph_j: ConstantPH) -> float:
    """Log acceptance probability for a pH-REMD swap between two replicas."""
    n_i = num_titratable_protons(cph_i)
    n_j = num_titratable_protons(cph_j)
    ph_i = cph_i.ph[cph_i.currentPHIndex]
    ph_j = cph_j.ph[cph_j.currentPHIndex]
    return (n_i - n_j) * np.log(10.0) * (ph_j - ph_i)


def _protonation_snapshot(cph: ConstantPH) -> dict[int, int]:
    return {res_index: t.current_index for res_index, t in cph.titrations.items()}


def _apply_protonation_snapshot(
    cph: ConstantPH,
    snapshot: dict[int, int],
) -> None:
    for res_index, state_index in snapshot.items():
        cph.set_residue_state(res_index, state_index, relax=False)


def _sync_auxiliary_contexts(cph: ConstantPH) -> None:
    """Align implicit and relaxation contexts with the explicit simulation."""
    state = cph.simulation.context.getState(positions=True, parameters=True)
    explicit_positions = state.getPositions(asNumpy=True).value_in_unit(nanometers)
    cph.implicitContext.setPositions(explicit_positions[cph.implicitAtomIndex])
    cph.relaxationContext.setPositions(explicit_positions)
    cph.relaxationContext.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    sim_params = state.getParameters()
    for param in cph.relaxationContext.getParameters():
        cph.relaxationContext.setParameter(param, sim_params[param])


def _apply_md_state(cph: ConstantPH, md_state: State) -> None:
    """Restore positions, velocities, and box vectors on a replica context."""
    cph.simulation.context.setPositions(md_state.getPositions())
    cph.simulation.context.setVelocities(md_state.getVelocities())
    cph.simulation.context.setPeriodicBoxVectors(*md_state.getPeriodicBoxVectors())


def _capture_md_state(cph: ConstantPH) -> State:
    """Capture a replica's positions, velocities, and periodic box as an OpenMM State."""
    return cph.simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        enforcePeriodicBox=True,
    )


class ConstantPHRemd:
    """pH replica exchange with per-replica simulated tempering.

    Parameters
    ----------
    topology, positions, config, references, titratable_residue_indices
        Forwarded to each underlying :class:`ConstantPH` replica.
    pH
        The full pH ladder shared by every replica.  Each replica may walk
        this ladder via simulated tempering during :meth:`attemptMCStep`.
    n_replicas
        Number of replicas.  A single replica (``n_replicas=1``) recovers
        standard simulated-tempering CpHMD.  Two replicas are sufficient for
        efficient pH-space mixing when replica exchange is enabled.
    initial_ph_indices
        Optional explicit starting indices into ``pH`` for each replica.
        When omitted, replicas are spread evenly across the ladder endpoints
        (replica 0 at the lowest pH, replica ``n_replicas - 1`` at the
        highest).
    weights
        Simulated-tempering weights for the pH ladder.  ``None`` triggers
        Wang-Landau auto-tuning independently on each replica until
        :meth:`set_weights` is called.
    ligand_variant_molecules, ring_flip_angles, platform, properties
        Forwarded to each :class:`ConstantPH` replica.
    """

    def __init__(
        self,
        topology: Topology,
        positions: unit.Quantity,
        ph: Sequence[float],
        config: "ConstantpHSettings",
        references: "dict[str, TitratableResidueReference]",
        titratable_residue_indices: Iterable[int],
        *,
        n_replicas: int = 1,
        initial_ph_indices: Sequence[int] | None = None,
        ligand_variant_molecules: "list[Molecule] | None" = None,
        ring_flip_angles: Sequence[float] | None = None,
        weights: list[float] | None = None,
        initial_variant_indices: "dict[int, int] | list[dict[int, int]] | None" = None,
        platform: Platform | None = None,
        properties: dict | None = None,
    ) -> None:
        if n_replicas < 1:
            raise ValueError("n_replicas must be at least 1")
        # Per-replica starting variants: a single dict is shared by all replicas;
        # a list assigns one dict per replica (diversified initial conditions).
        if isinstance(initial_variant_indices, list) and len(initial_variant_indices) != n_replicas:
            raise ValueError(
                f"initial_variant_indices list has length {len(initial_variant_indices)} "
                f"but n_replicas is {n_replicas}"
            )

        self.ph = [float(x) for x in ph]
        self.n_replicas = n_replicas
        self.replicas: list[ConstantPH] = []

        if initial_ph_indices is None:
            if n_replicas == 1:
                initial_ph_indices = [0]
            else:
                initial_ph_indices = [
                    round(i * (len(self.ph) - 1) / (n_replicas - 1)) for i in range(n_replicas)
                ]
        if len(initial_ph_indices) != n_replicas:
            raise ValueError(
                f"initial_ph_indices has length {len(initial_ph_indices)} "
                f"but n_replicas is {n_replicas}"
            )
        for index in initial_ph_indices:
            if index < 0 or index >= len(self.ph):
                raise ValueError(
                    f"initial pH index {index} is outside ladder range [0, {len(self.ph) - 1}]"
                )

        for replica_index, ph_index in enumerate(initial_ph_indices):
            replica_variant_indices = (
                initial_variant_indices[replica_index]
                if isinstance(initial_variant_indices, list)
                else initial_variant_indices
            )
            replica = ConstantPH(
                topology=topology,
                positions=positions,
                ph=self.ph,
                config=config,
                references=references,
                titratable_residue_indices=titratable_residue_indices,
                ligand_variant_molecules=ligand_variant_molecules,
                ring_flip_angles=ring_flip_angles,
                weights=weights,
                initial_variant_indices=replica_variant_indices,
                platform=platform,
                properties=properties,
            )
            replica.currentPHIndex = ph_index
            self.replicas.append(replica)

        self.n_exchange_attempts = 0
        self.n_exchange_accepted = 0

    @property
    def exchange_acceptance_rate(self) -> float:
        """Fraction of attempted replica exchanges that were accepted."""
        if self.n_exchange_attempts == 0:
            return 0.0
        return self.n_exchange_accepted / self.n_exchange_attempts

    def set_weights(self, weights: Sequence[float]) -> None:
        """Apply fixed simulated-tempering weights to every replica."""
        weight_list = list(weights)
        for replica in self.replicas:
            replica.set_ph(self.ph, weight_list)

    @property
    def weights(self) -> list[float]:
        """Simulated-tempering weights from replica 0."""
        return self.replicas[0].weights

    def step(self, steps: int) -> None:
        """Advance every replica by ``steps`` MD integration steps."""
        for replica in self.replicas:
            replica.simulation.step(steps)

    def attempt_mc_step(self) -> list[bool]:
        """Run one CpH MC step on each replica (includes simulated tempering)."""
        return [replica.attempt_mc_step() for replica in self.replicas]

    def attempt_replica_exchange(self, i: int = 0, j: int = 1) -> bool:
        """Attempt a pH-REMD exchange between replicas ``i`` and ``j``.

        Swaps coordinates, velocities, protonation states, and active pH
        indices when the Metropolis criterion is satisfied.
        """
        if i < 0 or j < 0 or i >= self.n_replicas or j >= self.n_replicas:
            raise IndexError(f"replica indices ({i}, {j}) out of range")
        if i == j:
            return False

        rep_i = self.replicas[i]
        rep_j = self.replicas[j]
        self.n_exchange_attempts += 1

        log_prob = replica_exchange_log_probability(rep_i, rep_j)
        if log_prob < 0.0 and np.random.random() >= np.exp(log_prob):
            return False

        self._swap_replica_states(rep_i, rep_j)
        self.n_exchange_accepted += 1
        return True

    def attempt_adjacent_exchanges(self) -> list[bool]:
        """Attempt exchanges between every adjacent replica pair."""
        results: list[bool] = []
        for pair in range(self.n_replicas - 1):
            results.append(self.attempt_replica_exchange(pair, pair + 1))
        return results

    @staticmethod
    def _swap_replica_states(rep_a: ConstantPH, rep_b: ConstantPH) -> None:
        protonation_a = _protonation_snapshot(rep_a)
        protonation_b = _protonation_snapshot(rep_b)
        ph_index_a = rep_a.currentPHIndex
        ph_index_b = rep_b.currentPHIndex

        md_state_a = _capture_md_state(rep_a)
        md_state_b = _capture_md_state(rep_b)

        _apply_protonation_snapshot(rep_a, protonation_b)
        _apply_protonation_snapshot(rep_b, protonation_a)

        _apply_md_state(rep_a, md_state_b)
        _apply_md_state(rep_b, md_state_a)

        rep_a.currentPHIndex = ph_index_b
        rep_b.currentPHIndex = ph_index_a

        _sync_auxiliary_contexts(rep_a)
        _sync_auxiliary_contexts(rep_b)

    def reset_stats(self) -> None:
        """Zero MC counters on every replica and exchange statistics."""
        for replica in self.replicas:
            replica.reset_stats()
        self.n_exchange_attempts = 0
        self.n_exchange_accepted = 0

    def current_ph_values(self) -> list[float]:
        """Return the currently active pH value for each replica."""
        return [replica.ph[replica.currentPHIndex] for replica in self.replicas]

    def replica_state_vectors(self) -> list[tuple[float, ...]]:
        """Return ``(pH, *protonation_indices)`` tuples for each replica."""
        titratable_indices = sorted(self.replicas[0].titrations)
        vectors: list[tuple[float, ...]] = []
        for replica in self.replicas:
            vectors.append(
                (
                    replica.ph[replica.currentPHIndex],
                    *(replica.titrations[index].current_index for index in titratable_indices),
                )
            )
        return vectors

    def save_checkpoint(self, directory: Path) -> None:
        """Persist OpenMM checkpoints and CpH/REMD state for every replica."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        replica_states = []
        for i, replica in enumerate(self.replicas):
            chk_path = directory / f"replica_{i}.chk"
            chk_path.write_bytes(replica.simulation.context.createCheckpoint())
            replica_states.append(
                {
                    "current_ph_index": replica.currentPHIndex,
                    "protonation": {
                        str(res_index): state_index
                        for res_index, state_index in _protonation_snapshot(replica).items()
                    },
                    "current_step": replica.simulation.currentStep,
                }
            )

        remd_payload = {
            "replicas": replica_states,
            "weights": self.weights,
            "n_exchange_attempts": self.n_exchange_attempts,
            "n_exchange_accepted": self.n_exchange_accepted,
        }
        (directory / "remd_state.json").write_text(json.dumps(remd_payload, indent=2))

    def load_checkpoint(self, directory: Path) -> None:
        """Restore replica MD and CpH state from :meth:`save_checkpoint`."""
        directory = Path(directory)
        remd_path = directory / "remd_state.json"
        if not remd_path.exists():
            raise FileNotFoundError(f"REMD checkpoint not found: {remd_path}")

        remd_payload = json.loads(remd_path.read_text())
        if len(remd_payload["replicas"]) != self.n_replicas:
            raise ValueError(
                f"Checkpoint has {len(remd_payload['replicas'])} replicas "
                f"but REMD has {self.n_replicas}"
            )

        self.set_weights(remd_payload["weights"])
        for i, rep_state in enumerate(remd_payload["replicas"]):
            replica = self.replicas[i]
            chk_path = directory / f"replica_{i}.chk"
            if not chk_path.exists():
                raise FileNotFoundError(f"Replica checkpoint not found: {chk_path}")

            replica.simulation.context.loadCheckpoint(chk_path.read_bytes())
            protonation = {
                int(res_index): state_index
                for res_index, state_index in rep_state["protonation"].items()
            }
            _apply_protonation_snapshot(replica, protonation)
            replica.currentPHIndex = rep_state["current_ph_index"]
            _sync_auxiliary_contexts(replica)

        self.n_exchange_attempts = remd_payload.get("n_exchange_attempts", 0)
        self.n_exchange_accepted = remd_payload.get("n_exchange_accepted", 0)
