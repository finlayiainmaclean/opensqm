"""Well-tempered metadynamics on the ligand RMSD, and its c(t) bias offset.

The collective variable is the heavy-atom RMSD of the ligand from its docked
pose, measured after superimposing on the protein. PLUMED expresses that by
splitting the reference PDB into alignment weights (protein) and displacement
weights (ligand). A single ``RMSDForce`` cannot: it superimposes whatever it
measures, so on the ligand alone it is blind to a ligand that leaves the pocket
rigidly -- translate a ligand a full nanometre and it still scores zero.

Two of them recover the protein-aligned value. Fitting protein and ligand
together, and the protein alone, differ only by the ligand's displacement in the
protein frame:

    n_tot RMSD_tot^2 - n_prot RMSD_prot^2 = (n_lig n_prot / n_tot) d^2

The n_prot/n_tot factor is there because the two fits use different centroids.
The identity is exact while the protein dominates the joint fit, which any real
protein does: against a direct Kabsch alignment the error is below 1e-4 nm for
2000 protein heavy atoms against 25 ligand ones, and it is exactly zero when the
whole complex translates or rotates. Nothing here depends on the protein being
restrained.

c(t) is the time-dependent offset of Tiwary and Parrinello (*JPCB* 2015), what
PLUMED's ``METAD ... CALC_RCT`` prints as ``metad.rct``:

    c(t) = (1/beta) ln [ int ds e^{beta gamma/(gamma-1) V(s,t)}
                       / int ds e^{beta 1/(gamma-1) V(s,t)} ]

It is a functional of the whole bias surface at time ``t``, not of the bias the
walker felt, so it is computed from the grid ``Metadynamics`` already keeps
rather than logged along the trajectory. PLUMED refreshes it every
``RCT_USTRIDE`` hills; here it is recomputed every frame, which is strictly
more current and costs two 200-point reductions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from openmm import CustomCentroidBondForce, CustomCVForce, RMSDForce, System, unit
from openmm.app import DCDReporter
from openmm.app.metadynamics import BiasVariable, Metadynamics
from scipy.special import logsumexp
from tqdm import tqdm

from opensqm.modbind.escape import build_simulation

if TYPE_CHECKING:
    from opensqm.ctmd.config import CTMDSettings
    from opensqm.modbind.states import PreparedState

# Residues that are neither the ligand nor part of the alignment frame.
NON_PROTEIN_RESIDUES = frozenset({"LIG", "HOH", "SOL", "WAT", "NA", "CL", "MG", "K", "ZN"})

# Floor inside the square root. Both RMSDs vanish at the reference pose, where
# d(sqrt)/du diverges; without it the very first step can produce a NaN force.
# It offsets the CV by 1e-4 nm, a hundredth of the hill width.
_SQRT_FLOOR = 1e-8


@dataclass
class CTMDTrajectory:
    """One replica's collective variable and bias offset, sampled every frame."""

    rmsd_nm: np.ndarray
    ct_kj: np.ndarray
    frame_interval_ps: float

    @property
    def duration_ns(self) -> float:
        """Simulated time of this replica."""
        return (len(self.rmsd_nm) - 1) * self.frame_interval_ps / 1000.0


def ct_from_bias(bias_kj: np.ndarray, kt_kj: float, bias_factor: float) -> float:
    """c(t) in kJ/mol for a bias ``V(s,t)`` tabulated on a uniform grid.

    ``bias_kj`` is V(s,t) in kJ/mol over the collective-variable grid, ``kt_kj``
    is kT in kJ/mol and ``bias_factor`` is the well-tempered gamma. The grid
    spacing cancels in the ratio, so only the values matter. A uniform bias of
    height V returns exactly V.
    """
    beta_v = np.ravel(np.asarray(bias_kj, dtype=np.float64)) / kt_kj
    scale = bias_factor / (bias_factor - 1.0)
    return kt_kj * float(logsumexp(beta_v * scale) - logsumexp(beta_v / (bias_factor - 1.0)))


def ct(meta: Metadynamics) -> float:
    """c(t) of a running metadynamics, in kJ/mol. Equals PLUMED's ``metad.rct``."""
    gamma = meta.biasFactor
    kt_kj = (unit.MOLAR_GAS_CONSTANT_R * meta.temperature).value_in_unit(unit.kilojoule_per_mole)
    # getFreeEnergy() returns -(gamma/(gamma-1)) V(s,t); invert it to recover V.
    free_energy = meta.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole)
    return ct_from_bias(-free_energy * (gamma - 1.0) / gamma, kt_kj, gamma)


def alignment_atoms(state: PreparedState) -> list[int]:
    """Heavy atoms of the protein, which the ligand RMSD is measured against."""
    return [
        atom.index
        for atom in state.topology.atoms()
        if atom.element is not None
        and atom.element.symbol != "H"
        and atom.residue.name not in NON_PROTEIN_RESIDUES
    ]


def ligand_rmsd_force(state: PreparedState) -> CustomCVForce:
    """Build the protein-aligned ligand RMSD in nm. See the module docstring.

    Each RMSDForce takes reference coordinates for every particle in the system
    even though it scores only its own subset.
    """
    protein = alignment_atoms(state)
    n_prot, n_lig = len(protein), len(state.ligand_indices)
    n_tot = n_prot + n_lig
    if n_prot < n_lig:
        raise ValueError(
            f"The alignment frame ({n_prot} protein heavy atoms) must outnumber the "
            f"ligand ({n_lig}); the joint fit would otherwise follow the ligand."
        )
    force = CustomCVForce(
        f"sqrt({_SQRT_FLOOR} + max(0, "
        f"({n_tot}*tot^2 - {n_prot}*prot^2)*{n_tot}/({n_lig}*{n_prot})))"
    )
    force.addCollectiveVariable("tot", RMSDForce(state.positions, protein + state.ligand_indices))
    force.addCollectiveVariable("prot", RMSDForce(state.positions, protein))
    return force


def image_ligand_with_protein(system: System, state: PreparedState) -> CustomCentroidBondForce:
    """Bond the ligand to the protein at zero strength, and add it to ``system``.

    RMSDForce ignores periodic boundaries, so it breaks the moment the ligand is
    imaged to the far side of the box. OpenMM works its molecules out from
    bonded interactions, so a zero-strength bond puts the ligand and the protein
    in one molecule and they are imaged together. The force contributes no
    energy and no gradient. Taken from OpenBPMD, which needs it for the same
    reason.
    """
    force = CustomCentroidBondForce(2, "0*distance(g1,g2)")
    force.addGroup(list(state.ligand_indices))
    force.addGroup(alignment_atoms(state))
    force.addBond([0, 1])
    force.setUsesPeriodicBoundaryConditions(True)
    system.addForce(force)
    return force


def build_metadynamics(
    state: PreparedState, config: CTMDSettings
) -> tuple[PreparedState, Metadynamics]:
    """Add the RMSD bias to a copy of ``state``'s system.

    Returns the copied state and the ``Metadynamics`` driving it. The copy
    matters: ``Metadynamics.__init__`` calls ``System.addForce``, and the
    prepared state's system is cached to XML and reused across replicas, so
    biasing it in place would poison that cache.
    """
    system = copy.deepcopy(state.system)
    image_ligand_with_protein(system, state)
    variable = BiasVariable(
        ligand_rmsd_force(state),
        minValue=0.0,
        maxValue=config.grid_max.value_in_unit(unit.nanometer),
        biasWidth=config.hill_sigma.value_in_unit(unit.nanometer),
        gridWidth=config.grid_bins,
    )
    meta = Metadynamics(
        system,
        [variable],
        config.temperature,
        config.bias_factor,
        config.hill_height,
        config.hill_interval_steps,
    )
    return replace(state, system=system), meta


def run_ctmd_replica(
    state: PreparedState,
    config: CTMDSettings,
    *,
    seed: int,
    dcd_path: str | None = None,
) -> CTMDTrajectory:
    """Run one metadynamics replica until the ligand commits to leaving.

    Records the RMSD and c(t) every ``frame_interval``, starting from the
    unbiased frame at t = 0, and stops early once the RMSD has held above
    ``commit_cutoff`` for a full ``commit_time`` window or ``max_time`` is
    reached. The scored frame is re-derived from ``rmsd_nm`` by
    :func:`opensqm.ctmd.score.commit_frame`, so the counter below only ends the
    run; it never sets the score.
    """
    biased_state, meta = build_metadynamics(state, config)
    simulation = build_simulation(
        biased_state,
        temperature=config.temperature,
        step_size=config.integrator_step_size,
        friction=config.friction,
        seed=seed,
    )
    if dcd_path is not None:
        simulation.reporters.append(DCDReporter(dcd_path, config.frame_interval_steps))

    cutoff = config.commit_cutoff.value_in_unit(unit.nanometer)
    rmsds = [float(meta.getCollectiveVariables(simulation)[0])]
    cts = [ct(meta)]
    above = 0
    for _ in tqdm(range(config.max_frames), desc="CTMD replica", unit="frame"):
        meta.step(simulation, config.frame_interval_steps)
        rmsd = float(meta.getCollectiveVariables(simulation)[0])
        rmsds.append(rmsd)
        cts.append(ct(meta))
        above = above + 1 if rmsd > cutoff else 0
        if above >= config.commit_frames:
            break

    trajectory = CTMDTrajectory(
        rmsd_nm=np.asarray(rmsds, dtype=np.float64),
        ct_kj=np.asarray(cts, dtype=np.float64),
        frame_interval_ps=config.frame_interval.value_in_unit(unit.picosecond),
    )
    logger.info(
        f"Replica ran {trajectory.duration_ns:.3f} ns, "
        f"final RMSD {rmsds[-1]:.2f} nm, c(t) {cts[-1]:.1f} kJ/mol"
    )
    return trajectory
