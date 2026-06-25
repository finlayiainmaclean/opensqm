from collections import defaultdict
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from openmm import Context, GBSAOBCForce, NonbondedForce
from openmm.app import Modeller, Simulation, element
from openmm.app.forcefield import NonbondedGenerator
from openmm.app.internal import compiled
from openmm.unit import MOLAR_GAS_CONSTANT_R, elementary_charge, nanometers
from openmm.unit import sum as unitsum
from rdkit import Chem
from rdsl import select_atom_ids

from opensqm.md.omm import (
    map_pdb_residue_keys_to_openmm_indices,
    pdb_residue_key_from_rdkit,
)
from opensqm.md.terminal_ring_mc import (
    TerminalGroup,
    find_terminal_group,
)

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule  # type: ignore

    from opensqm.cph.reference_energy.models import TitratableResidueReference
    from opensqm.cph.simulation_config import ConstantpHSettings


# Default candidate-rotation pool for terminal-group MC. Chosen to mirror
# the previous run.py defaults (30..210 deg in 30 deg steps); the per-move
# sign is randomised at sample time.
# DEFAULT_RING_FLIP_ANGLES: tuple[float, ...] = tuple(
#     float(a) for a in np.arange(30.0, 210.0, 30.0)
# )
DEFAULT_RING_FLIP_ANGLES: tuple[float, ...] = (180.0,)


@dataclass
class _RingFlipRecord:
    """Per-residue terminal-group rotation move, resolved against ConstantPH's
    internal explicit and implicit topologies (both of which have had
    ``Modeller.addHydrogens`` called on them).
    """

    angles: list[float]
    explicit: TerminalGroup
    implicit: TerminalGroup


class ResidueState(object):
    def __init__(self, residueIndex, atomIndices, particleParameters, exceptionParameters, numHydrogens):
        self.residueIndex = residueIndex
        self.atomIndices = atomIndices
        self.particleParameters = particleParameters
        self.exceptionParameters = exceptionParameters
        self.numHydrogens = numHydrogens


@dataclass
class ResidueTitration:
    """Dynamic, per-residue-index state for one titratable residue.

    Static information (variant names, formal charges, reference energies,
    transitions) lives on :attr:`reference`, which is a
    :class:`~opensqm.cph.reference_energy.models.TitratableResidueReference`
    shared across all topology residues of the same residue type. The
    fields here hold simulation state that is per-instance: the bound
    ``ResidueState`` objects, the currently-active variant index, and
    Monte-Carlo acceptance counters.

    ``referenceEnergies`` is materialised as a local mutable list copy of
    ``reference.reference_energies`` at construction so the
    :class:`~opensqm.cph.reference_energy.finder.ReferenceEnergyFinder`
    can refine it during pairwise fits without polluting the shared
    reference object.
    """

    reference: "TitratableResidueReference"
    referenceEnergies: list = field(default_factory=list)
    explicitStates: list = field(default_factory=list)
    implicitStates: list = field(default_factory=list)
    explicitHydrogenIndices: list = field(default_factory=list)
    protonatedIndex: int = -1
    currentIndex: int = -1
    n_state_attempts: int = 0
    n_state_accepted: int = 0
    n_standalone_flip_attempts: int = 0
    n_standalone_flip_accepted: int = 0
    n_coupled_flip_attempts: int = 0
    n_coupled_flip_accepted: int = 0

    def __post_init__(self):
        if not self.referenceEnergies:
            self.referenceEnergies = list(self.reference.reference_energies)

    @property
    def variants(self):
        """Per-state variants as accepted by ``Modeller.addHydrogens``."""
        return self.reference.variants

    @property
    def variant_names(self):
        """Per-state human-readable variant labels (e.g. ``["HIP", "HID", "HIE"]``)."""
        return self.reference.variant_names

    @property
    def charges(self):
        """Per-state formal charges (e.g. ``[1, 0, 0]`` for HIS)."""
        return self.reference.charges

    @property
    def currentCharge(self):
        """Formal charge of the currently-active variant, or ``None`` if unknown."""
        if self.currentIndex < 0:
            return None
        return self.charges[self.currentIndex]

    @property
    def state_acceptance_rate(self):
        """Fraction of state-swap proposals touching this residue that were accepted."""
        if self.n_state_attempts == 0:
            return 0.0
        return self.n_state_accepted / self.n_state_attempts

    @property
    def standalone_flip_acceptance_rate(self):
        """Fraction of stand-alone ring-flip proposals on this residue that were accepted."""
        if self.n_standalone_flip_attempts == 0:
            return 0.0
        return self.n_standalone_flip_accepted / self.n_standalone_flip_attempts

    @property
    def coupled_flip_acceptance_rate(self):
        """Fraction of (state-swap + flip) coupled proposals on this residue that were accepted."""
        if self.n_coupled_flip_attempts == 0:
            return 0.0
        return self.n_coupled_flip_accepted / self.n_coupled_flip_attempts

    def reset_stats(self):
        """Zero all MC counters for this residue."""
        self.n_state_attempts = 0
        self.n_state_accepted = 0
        self.n_standalone_flip_attempts = 0
        self.n_standalone_flip_accepted = 0
        self.n_coupled_flip_attempts = 0
        self.n_coupled_flip_accepted = 0


def _min_image_distance_matrix(
    positions_a_nm: np.ndarray,
    positions_b_nm: np.ndarray,
    box_vectors_nm: np.ndarray | None,
) -> np.ndarray:
    """Pairwise distances under (optional) periodic boundary conditions.

    ``box_vectors_nm`` is either ``None`` (non-periodic) or a ``(3, 3)``
    array of OpenMM reduced-form box vectors (rows ``a``, ``b``, ``c`` with
    ``a`` along x and ``b`` in the xy plane). Triclinic reduction is applied
    in the standard ``c → b → a`` order, which gives the correct minimum
    image for any box satisfying OpenMM's reduced-form constraints
    (``a.x > 2|b.x|``, ``a.x > 2|c.x|``, ``b.y > 2|c.y|``).
    """
    diffs = positions_a_nm[:, None, :] - positions_b_nm[None, :, :]
    if box_vectors_nm is not None:
        a, b, c = box_vectors_nm[0], box_vectors_nm[1], box_vectors_nm[2]
        diffs -= c * np.round(diffs[..., 2:3] / c[2])
        diffs -= b * np.round(diffs[..., 1:2] / b[1])
        diffs -= a * np.round(diffs[..., 0:1] / a[0])
    return np.sqrt((diffs ** 2).sum(-1))


def select_titratable_residues(
    topology,
    positions,
    references: "dict[str, TitratableResidueReference]",
    ligand_residue_name: str | Sequence[str] | None = None,
    cutoff=None,
) -> list[int]:
    """Pick the titratable residue indices to drive in a complex simulation.

    A residue is included if its name appears in ``references`` and either:

    * the residue's name equals one of the ``ligand_residue_name`` entries
      (small-molecule cofactors/ligands are always titrated when listed), or
    * ``cutoff`` is ``None`` (no spatial gating), or
    * the residue lies within ``cutoff`` of any ligand atom, measured as
      a heavy-atom minimum-image distance in the *input* positions. When
      ``topology.getPeriodicBoxVectors()`` is set (the usual case for a
      solvated complex) the minimum-image convention is applied using
      those box vectors, so the result is invariant to molecule wrapping
      from e.g. NPT equilibration.

    Parameters
    ----------
    topology : openmm.app.Topology
        The fully-built complex topology (usually the post-``solvate_ligand`` /
        ``prepare_complex`` output). Its periodic box vectors, if any, are
        used for the minimum-image distance computation.
    positions : sequence of openmm.unit.Quantity-like
        Positions aligned with ``topology``.
    references : dict[str, TitratableResidueReference]
        Output of :func:`opensqm.cph.reference_energy.generate.generate_all`.
    ligand_residue_name : str or sequence of str, optional
        Residue name(s) of small-molecule ligands/cofactors (e.g. ``"LIG"``,
        ``"COF"``, or the OpenFF ``Molecule.name``). Required when
        ``cutoff`` is not ``None`` so the cutoff has reference atom sets to
        measure against. Pass ``None`` together with ``cutoff=None`` to
        titrate every residue whose name is in ``references`` regardless of
        position.
    cutoff : openmm.unit.Quantity, optional
        Distance cutoff (e.g. ``5.0 * unit.angstrom``). Residues whose
        minimum heavy-atom distance to the ligand exceeds ``cutoff`` are
        excluded. Pass ``None`` to disable the gate.

    Returns
    -------
    list[int]
        Residue indices in ascending order, suitable for passing as
        ``titratable_residue_indices`` to :class:`ConstantPH`.
    """
    from openmm import unit  # local import to avoid hard dep at module load

    if cutoff is None:
        return sorted(
            r.index for r in topology.residues() if r.name in references
        )

    if ligand_residue_name is None:
        raise ValueError(
            "ligand_residue_name is required when cutoff is not None"
        )

    if isinstance(ligand_residue_name, str):
        anchor_residue_names = [ligand_residue_name]
    else:
        anchor_residue_names = list(ligand_residue_name)
    if not anchor_residue_names:
        raise ValueError(
            "ligand_residue_name is required when cutoff is not None"
        )

    cutoff_nm = float(cutoff.value_in_unit(unit.nanometer))
    if hasattr(positions, "value_in_unit"):
        all_positions_nm = np.asarray(positions.value_in_unit(unit.nanometer))
    else:
        all_positions_nm = np.asarray(positions)

    box = topology.getPeriodicBoxVectors()
    if box is None:
        box_vectors_nm: np.ndarray | None = None
    else:
        box_vectors_nm = np.array(
            [[float(v.value_in_unit(unit.nanometer)) for v in row] for row in box]
        )
        # Reduced-form sanity check: diagonal must be positive. A degenerate
        # box (zero-length vector) would make minimum-image undefined.
        if not (box_vectors_nm[0, 0] > 0
                and box_vectors_nm[1, 1] > 0
                and box_vectors_nm[2, 2] > 0):
            raise ValueError(
                "Periodic box vectors are degenerate; cannot compute "
                f"minimum-image distances. Got {box_vectors_nm!r}"
            )

    ligand_atom_indices = [
        atom.index
        for residue in topology.residues()
        if residue.name in anchor_residue_names
        for atom in residue.atoms()
    ]
    if not ligand_atom_indices:
        raise RuntimeError(
            f"No residues named {anchor_residue_names!r} found in topology"
        )
    ligand_positions_nm = all_positions_nm[ligand_atom_indices]

    selected: list[int] = []
    for residue in topology.residues():
        if residue.name not in references:
            continue
        if residue.name in anchor_residue_names:
            selected.append(residue.index)
            continue
        residue_atom_indices = [atom.index for atom in residue.atoms()]
        residue_positions_nm = all_positions_nm[residue_atom_indices]
        dists = _min_image_distance_matrix(
            residue_positions_nm, ligand_positions_nm, box_vectors_nm
        )
        if float(dists.min()) <= cutoff_nm:
            selected.append(residue.index)
    return sorted(selected)


def select_titratable_residues_by_rdsl(
    topology,
    pdb_path,
    references: "dict[str, TitratableResidueReference]",
    query: str,
) -> list[int]:
    """Pick titratable residue indices by intersecting references with an RDSL query.

    All residues whose name appears in ``references`` are candidates. The final
    selection is the subset whose atoms match ``query`` on the PDB structure
    (loaded into RDKit with residue IDs preserved).
    """
    cocomplex = Chem.MolFromPDBFile(
        str(Path(pdb_path)), sanitize=False, removeHs=False
    )
    if cocomplex is None:
        raise RuntimeError(f"Failed to load PDB for RDSL selection: {pdb_path}")

    candidate_indices = {
        residue.index
        for residue in topology.residues()
        if residue.name in references
    }
    if not candidate_indices:
        return []

    selected_atom_ids = select_atom_ids(cocomplex, query)
    selected_residue_keys = {
        key
        for atom_idx in selected_atom_ids
        if (key := pdb_residue_key_from_rdkit(cocomplex.GetAtomWithIdx(int(atom_idx))))
        is not None
    }
    selected_residue_indices = map_pdb_residue_keys_to_openmm_indices(
        topology, selected_residue_keys
    )
    return sorted(candidate_indices & selected_residue_indices)




def _disulfide_bonded_residue_indices(topology) -> set[int]:
    """Indices of CYS residues whose SG is bonded to another residue's SG.

    Disulfide cysteines are oxidized (CYX): the sulfur carries no titratable
    proton, and the extra S-S external bond means the free-thiol CYS template
    cannot be matched by ``Modeller.addHydrogens``. Such residues must be
    excluded from titration.
    """
    disulfide: set[int] = set()
    for bond in topology.bonds():
        a1, a2 = bond[0], bond[1]
        if (
            a1.element == element.sulfur
            and a2.element == element.sulfur
            and a1.residue.index != a2.residue.index
        ):
            disulfide.add(a1.residue.index)
            disulfide.add(a2.residue.index)
    return disulfide


def select_titratable_residue_indices(
    topology,
    pdb_path: Path | str,
    references: "dict[str, TitratableResidueReference]",
    query: str | None = None,
) -> list[int]:
    """Pick titratable residue indices for a constant-pH run.

    Without ``query``, every topology residue whose name appears in
    ``references`` is selected. With ``query``, the RDSL expression is
    intersected with those candidates.

    Disulfide-bonded cysteines (SG covalently bonded to another SG) are always
    excluded: they are oxidized and have no titratable proton.

    When ``small_molecule_setups`` is provided, ligand/cofactor residues are
    dropped if they have only one protomer template and no ring-flip MC, and
    multi-protomer small molecules are kept or added when they match
    ``query``. Alternate protomers such as ``LIG1`` are MC templates for the
    single ``LIG`` topology residue, not separate indices.
    """
    if query is None:
        titratable = {
            residue.index
            for residue in topology.residues()
            if residue.name in references
        }
    else:
        titratable = set(
            select_titratable_residues_by_rdsl(topology, pdb_path, references, query)
        )

    titratable -= _disulfide_bonded_residue_indices(topology)

    return sorted(titratable)


class ConstantPH(object):
    """
    Construct a ConstantPH object that can be used to run a simulation at constant pH.

    Parameters
    ----------
    topology : openmm.app.Topology
        The model to simulate. If a residue can exist in multiple protonation
        states, this Topology may use any one of them. The alternate versions
        will be constructed by calling ``Modeller.addHydrogens()`` against the
        relevant variant entries in ``references``.
    positions : list
        The initial positions of the atoms.
    pH : float or list[float]
        The pH to perform the simulation at. A single number runs at a fixed
        pH; a list of numbers turns on simulated tempering across them.
    config : SimulationConfig
        Simulation configuration. Used both to build the explicit/implicit
        forcefields (with the optional ``ligand_variant_molecules`` registered
        as SMIRNOFF templates) and to pull integrator/system kwargs.
    references : dict[str, TitratableResidueReference]
        Static, per-residue-type definitions: variant names, hydrogen
        layouts, formal charges, reference energies, and the microscopic
        transitions. Keyed by ``TitratableResidueReference.residue_name``
        (e.g. ``"HIS"`` or the ligand's ``Molecule.name``). Typically the
        output of :func:`opensqm.cph.reference_energy.generate.generate_all`.
    titratable_residue_indices : Iterable[int]
        Topology residue indices to drive with constant-pH MC. Each must
        refer to a residue whose ``residue.name`` is a key of
        ``references``. Use :func:`select_titratable_residue_indices`,
        :func:`select_titratable_residues_by_rdsl`, or
        :func:`select_titratable_residues` to pick this
        list (e.g. by distance to a ligand).
    ligand_variant_molecules : list[openff.toolkit.topology.Molecule], optional
        Forwarded to ``config.get_explicit_forcefield`` /
        ``config.get_implicit_forcefield`` so SMIRNOFF templates for the
        ligand protomers are registered on the constructed forcefields.
        Pass the same variant list used to compute the reference energies.
        ``None`` skips ligand template registration (protein-only systems).
    ring_flip_angles : Sequence[float] or None, optional
        Candidate-rotation pool (in degrees) for terminal-group flips on
        any titratable residue whose
        :attr:`TitratableResidueReference.ring_flip_bonds` is non-empty.
        The per-move sign is randomised at sample time. Pass ``None`` to
        disable ring-flip MC entirely; the default mirrors the previous
        run.py choice of 30..210 deg in 30 deg steps.
    weights : list, optional
        The weight factor to use for each pH in the simulated tempering
        algorithm. ``None`` triggers Wang-Landau auto-tuning.
    platform : openmm.Platform, optional
    properties : dict, optional
        Platform-specific properties to pass to the Context's constructor.
    """

    def __init__(
        self,
        topology,
        positions,
        pH,
        config: "ConstantpHSettings",
        references: "dict[str, TitratableResidueReference]",
        titratable_residue_indices: Iterable[int],
        *,
        ligand_variant_molecules: "list[Molecule] | None" = None,
        ring_flip_angles: Sequence[float] | None = DEFAULT_RING_FLIP_ANGLES,
        weights=None,
        platform=None,
        properties=None,
    ):
        explicitForceField = config.get_explicit_forcefield(ligand_variant_molecules)
        implicitForceField = config.get_implicit_forcefield(ligand_variant_molecules)
        explicitArgs = config.explicit_params
        implicitArgs = config.implicit_params
        integrator = config.integrator
        relaxationIntegrator = config.relaxation_integrator
        self.config = config
        self.references = references
        if not isinstance(pH, Sequence):
            pH = [pH]
        self.setPH(pH, weights)
        self.currentPHIndex = 0
        self._explicitArgs = explicitArgs
        self._implicitArgs = implicitArgs
        self.relaxationSteps = config.relaxation_steps

        # Resolve the titratable-residue indices against the input topology
        # and build per-residue dynamic state from the shared references.
        residues_by_index = {r.index: r for r in topology.residues()}
        titratable_residue_indices = sorted(set(titratable_residue_indices))
        self.titrations: dict[int, ResidueTitration] = {}
        residueVariants: dict[int, list] = {}
        for resIndex in titratable_residue_indices:
            if resIndex not in residues_by_index:
                raise ValueError(
                    f"titratable residue index {resIndex} is not present in the topology"
                )
            residue = residues_by_index[resIndex]
            if residue.name not in references:
                raise ValueError(
                    f"residue {residue.name}.{resIndex} has no reference entry; "
                    f"available reference residue names: {sorted(references)}"
                )
            reference = references[residue.name]
            self.titrations[resIndex] = ResidueTitration(reference=reference)
            residueVariants[resIndex] = list(reference.variants)

        implicitToExplicitResidueMap = []
        explicitToImplicitResidueMap = {}
        solventResidues = []

        # Build the implicit solvent topology by removing water and ions.

        ionElements = (element.cesium, element.potassium, element.lithium, element.sodium, element.rubidium,
                       element.chlorine, element.bromine, element.fluorine, element.iodine)
        for residue in topology.residues():
            if residue.name == 'HOH' or (len(residue) == 1 and next(residue.atoms()).element in ionElements):
                solventResidues.append(residue)
            else:
                implicitToExplicitResidueMap.append(residue.index-len(solventResidues))
        for i, j in enumerate(implicitToExplicitResidueMap):
            explicitToImplicitResidueMap[j] = i
        modeller = Modeller(topology, positions)
        modeller.delete(solventResidues)
        implicitTopology = modeller.topology
        implicitPositions = modeller.positions

        # Loop over variants to construct a ResidueState for every variant of every titratable residue.

        # ``Modeller.addHydrogens`` expects ``variants[i]`` to either be
        # ``None`` (use the default residue template) or one of the
        # alternate variant names that the residue actually advertises
        # in ``Hydrogens.xml`` (e.g. ``HIP``/``HID``/``HIE`` for HIS).
        # Single-variant residues like ASN / GLN that we register
        # purely for ring-flip MC carry their own residue name as the
        # variant label (``variants=['ASN']`` for an ASN residue) -
        # passing that string through trips the ``Illegal variant``
        # check. Convert "self-named" string variants to ``None`` only
        # in that single-variant case so Modeller falls back to the
        # default template. Multi-variant residues whose main state
        # shares the PDB residue name (CYS/CYX, LYS/LYN) must keep the
        # explicit variant label so every protonation state is built.
        explicit_residue_names = {
            r.index: r.name for r in topology.residues()
        }
        implicit_residue_names = {
            r.index: r.name for r in implicitTopology.residues()
        }

        def _to_modeller_variant(variant, residue_name, *, single_variant: bool):
            # Only ring-flip-only residues (ASN/GLN) use their residue name
            # as the sole variant label. Passing that string to addHydrogens
            # trips OpenMM's "Illegal variant" check, so fall back to None.
            # Multi-variant residues whose main state shares the residue name
            # (CYS/CYX, LYS/LYN) must keep the explicit variant label.
            if single_variant and isinstance(variant, str) and variant == residue_name:
                return None
            return variant

        variantIndex = 0
        finished = False
        explicitVariants = [None]*topology.getNumResidues()
        implicitVariants = [None]*implicitTopology.getNumResidues()
        while not finished:
            finished = True

            # Build the explicit solvent states.

            active_residue_indices: set[int] = set()
            for resIndex, variants in residueVariants.items():
                if variantIndex < len(variants):
                    finished = False
                    active_residue_indices.add(resIndex)
                    explicitVariants[resIndex] = _to_modeller_variant(
                        variants[variantIndex],
                        explicit_residue_names[resIndex],
                        single_variant=len(variants) == 1,
                    )
            explicitStates = self._findResidueStates(
                topology, positions, explicitForceField, explicitVariants, explicitArgs,
                record_residue_indices=active_residue_indices,
            )

            # Build the implicit solvent states.

            active_implicit_indices: set[int] = set()
            for implicitIndex, explicitIndex in enumerate(implicitToExplicitResidueMap):
                if explicitIndex in residueVariants:
                    variants = residueVariants[explicitIndex]
                    if variantIndex < len(variants):
                        active_implicit_indices.add(implicitIndex)
                        implicitVariants[implicitIndex] = _to_modeller_variant(
                            variants[variantIndex],
                            implicit_residue_names[implicitIndex],
                            single_variant=len(variants) == 1,
                        )
            implicitStates = self._findResidueStates(
                implicitTopology, implicitPositions, implicitForceField, implicitVariants, implicitArgs,
                record_residue_indices=active_implicit_indices,
            )
            assert len(explicitStates) == len(implicitStates)

            # Add them to the ResidueTitration.

            for explicitState, implicitState in zip(explicitStates, implicitStates, strict=False):
                titration = self.titrations[explicitState.residueIndex]
                if variantIndex < len(titration.variants):
                    titration.explicitStates.append(explicitState)
                    titration.implicitStates.append(implicitState)
            variantIndex += 1

        # Create final versions of the topologies, including the fully protonated versions of all residues.

        for titration in self.titrations.values():
            titration.protonatedIndex = int(np.argmax([len(state.atomIndices) for state in titration.explicitStates]))
        variants = [None]*topology.getNumResidues()
        for resIndex in residueVariants:
            titration = self.titrations[resIndex]
            variants[resIndex] = _to_modeller_variant(
                titration.variants[titration.protonatedIndex],
                explicit_residue_names[resIndex],
                single_variant=len(titration.variants) == 1,
            )
        modeller = Modeller(topology, positions)
        modeller.addHydrogens(forcefield=explicitForceField, variants=variants)
        self.explicitTopology = modeller.topology
        explicitPositions = modeller.positions
        variants = [None]*implicitTopology.getNumResidues()
        for implicitIndex, explicitIndex in enumerate(implicitToExplicitResidueMap):
            if explicitIndex in residueVariants:
                titration = self.titrations[explicitIndex]
                variants[implicitIndex] = _to_modeller_variant(
                    titration.variants[titration.protonatedIndex],
                    implicit_residue_names[implicitIndex],
                    single_variant=len(titration.variants) == 1,
                )
        modeller = Modeller(implicitTopology, implicitPositions)
        modeller.addHydrogens(forcefield=implicitForceField, variants=variants)
        self.implicitTopology = modeller.topology
        implicitPositions = modeller.positions
        explicitResidues = list(self.explicitTopology.residues())
        implicitResidues = list(self.implicitTopology.residues())

        # Create systems for them.  Also create a third system that is identical to the explicit one,
        # but freezes non-solvent atoms.

        explicitSystem = explicitForceField.createSystem(self.explicitTopology, **explicitArgs)
        implicitSystem = implicitForceField.createSystem(self.implicitTopology, **implicitArgs)
        relaxationSystem = deepcopy(explicitSystem)
        for residue in self.explicitTopology.residues():
            if residue.name != 'HOH' and (len(residue) > 1 or next(residue.atoms()).element not in ionElements):
                for atom in residue.atoms():
                    relaxationSystem.setParticleMass(atom.index, 0.0)

        # For each ResidueTitration, identify the fully protonated state.  Replace the other states
        # with ones that include all protons, setting the parameters of the missing ones to 0.

        for resIndex, titration in self.titrations.items():
            protonated = titration.protonatedIndex
            titration.currentIndex = protonated
            explicitProtonatedParams = titration.explicitStates[protonated].particleParameters
            implicitProtonatedParams = titration.implicitStates[protonated].particleParameters
            explicitProtonatedExceptionParams = titration.explicitStates[protonated].exceptionParameters
            implicitProtonatedExceptionParams = titration.implicitStates[protonated].exceptionParameters
            explicitAtomIndices = {atom.name: atom.index for atom in explicitResidues[resIndex].atoms()}
            implicitAtomIndices = {atom.name: atom.index for atom in implicitResidues[explicitToImplicitResidueMap[resIndex]].atoms()}
            for i in range(len(titration.explicitStates)):
                if i != protonated:
                    oldExplicit = titration.explicitStates[i]
                    oldImplicit = titration.implicitStates[i]
                    newExplicit = deepcopy(titration.explicitStates[protonated])
                    newImplicit = deepcopy(titration.implicitStates[protonated])
                    newExplicit.numHydrogens = oldExplicit.numHydrogens
                    newImplicit.numHydrogens = oldImplicit.numHydrogens
                    for forceIndex in newExplicit.particleParameters:
                        params = oldExplicit.particleParameters[forceIndex]
                        for atomName in newExplicit.particleParameters[forceIndex]:
                            if atomName in params:
                                newExplicit.particleParameters[forceIndex][atomName] = params[atomName]
                            else:
                                newExplicit.particleParameters[forceIndex][atomName] = self._get_zero_parameters(explicitProtonatedParams[forceIndex][atomName], explicitSystem.getForce(forceIndex))
                                titration.explicitHydrogenIndices.append(explicitAtomIndices[atomName])
                    for forceIndex in newExplicit.exceptionParameters:
                        params = oldExplicit.exceptionParameters[forceIndex]
                        for key in newExplicit.exceptionParameters[forceIndex]:
                            if key in params:
                                newExplicit.exceptionParameters[forceIndex][key] = params[key]
                            else:
                                newExplicit.exceptionParameters[forceIndex][key] = [0.0, *list(explicitProtonatedExceptionParams[forceIndex][key][1:])]
                    for forceIndex in newImplicit.particleParameters:
                        params = oldImplicit.particleParameters[forceIndex]
                        for atomName in newImplicit.particleParameters[forceIndex]:
                            if atomName in params:
                                newImplicit.particleParameters[forceIndex][atomName] = params[atomName]
                            else:
                                newImplicit.particleParameters[forceIndex][atomName] = self._get_zero_parameters(implicitProtonatedParams[forceIndex][atomName], implicitSystem.getForce(forceIndex))
                    for forceIndex in newImplicit.exceptionParameters:
                        params = oldImplicit.exceptionParameters[forceIndex]
                        for key in newImplicit.exceptionParameters[forceIndex]:
                            if key in params:
                                newImplicit.exceptionParameters[forceIndex][key] = params[key]
                            else:
                                newImplicit.exceptionParameters[forceIndex][key] = [0.0, *list(implicitProtonatedExceptionParams[forceIndex][key][1:])]
                    titration.explicitStates[i] = newExplicit
                    titration.implicitStates[i] = newImplicit
            for i in range(len(titration.explicitStates)):
                titration.explicitStates[i].atomIndices = explicitAtomIndices
                titration.implicitStates[i].atomIndices = implicitAtomIndices

        # Create contexts or simulations for all the systems.

        self.simulation = Simulation(self.explicitTopology, explicitSystem, deepcopy(integrator), platform, properties)
        platform = self.simulation.context.getPlatform()
        if properties is None:
            self.implicitContext = Context(implicitSystem, deepcopy(integrator), platform)
            self.relaxationContext = Context(relaxationSystem, deepcopy(relaxationIntegrator), platform)
        else:
            self.implicitContext = Context(implicitSystem, deepcopy(integrator), platform, properties)
            self.relaxationContext = Context(relaxationSystem, deepcopy(relaxationIntegrator), platform, properties)
        self.simulation.context.setPositions(explicitPositions)
        self.relaxationContext.setPositions(explicitPositions)
        self.implicitContext.setPositions(implicitPositions)

        # Record the mapping from implicit system atoms to explicit system atoms.  We need this
        # for copying positions.

        implicitAtomIndex = [None]*implicitSystem.getNumParticles()
        for implicitIndex, explicitIndex in enumerate(implicitToExplicitResidueMap):
            explicitRes = explicitResidues[explicitIndex]
            implicitRes = implicitResidues[implicitIndex]
            explicitAtoms = {atom.name: atom.index for atom in explicitRes.atoms()}
            for atom in implicitRes.atoms():
                implicitAtomIndex[atom.index] = explicitAtoms[atom.name]
        self.implicitAtomIndex = np.array(implicitAtomIndex)

        # Record the indices of nonbonded exceptions in each of the contexts.

        self.explicitExceptionIndex = self._findExceptionIndices(explicitSystem, self.explicitTopology)
        self.implicitExceptionIndex = self._findExceptionIndices(implicitSystem, self.implicitTopology)
        self.explicitInterResidue14 = self._findInterResidue14(explicitSystem, self.explicitTopology)
        self.implicitInterResidue14 = self._findInterResidue14(implicitSystem, self.implicitTopology)

        # Record the scale factors for 1-4 Coulomb interactions.

        self.explicit14Scale = self._find14Scale(explicitForceField)
        self.implicit14Scale = self._find14Scale(implicitForceField)

        # Discover ring-flip MC moves on titratable residues whose
        # ``TitratableResidueReference.ring_flip_bonds`` is non-empty
        # (HIS carries its CB-CG flip by default; ligands carry whatever
        # bonds were autodetected by
        # :func:`opensqm.torsion_scanner.autodetect_flip_dihedrals_named`
        # at reference-generation time). One ``_RingFlipRecord`` is
        # produced per bond, so a residue with multiple rotatable bonds
        # gets multiple records. Atom indices are resolved against the
        # post-addHydrogens topologies, which is the geometry the MC
        # moves actually operate on.

        self.ringFlips: dict[int, list[_RingFlipRecord]] = {}
        if ring_flip_angles:
            angles_list = [float(a) for a in ring_flip_angles]
            for res_index, titration in self.titrations.items():
                bonds = titration.reference.ring_flip_bonds
                if not bonds:
                    continue
                explicit_residue = explicitResidues[res_index]
                explicit_atoms_by_name = {a.name: a.index for a in explicit_residue.atoms()}
                implicit_residue = implicitResidues[explicitToImplicitResidueMap[res_index]]
                implicit_atoms_by_name = {a.name: a.index for a in implicit_residue.atoms()}
                records: list[_RingFlipRecord] = []
                for anchor_name, pivot_name in bonds:
                    missing = [
                        n for n in (anchor_name, pivot_name)
                        if n not in explicit_atoms_by_name
                        or n not in implicit_atoms_by_name
                    ]
                    if missing:
                        raise ValueError(
                            f"ring-flip bond ({anchor_name!r}, {pivot_name!r}) "
                            f"on residue {explicit_residue.name}.{res_index} "
                            f"references atoms not present in topology: {missing}"
                        )
                    explicit_group = find_terminal_group(
                        self.explicitTopology,
                        explicit_atoms_by_name[anchor_name],
                        explicit_atoms_by_name[pivot_name],
                        angles=list(angles_list),
                    )
                    implicit_group = find_terminal_group(
                        self.implicitTopology,
                        implicit_atoms_by_name[anchor_name],
                        implicit_atoms_by_name[pivot_name],
                        angles=list(angles_list),
                    )
                    records.append(_RingFlipRecord(
                        angles=list(angles_list),
                        explicit=explicit_group,
                        implicit=implicit_group,
                    ))
                if records:
                    self.ringFlips[res_index] = records

        self.ringFlipAttempts = 0
        self.ringFlipAccepted = 0

        self.temperature = self.simulation.integrator.getTemperature()

    def setPH(self, pH, weights=None):
        """
        Set the pH to run the simulation at.  See the description of the `pH` and `weights` arguments to the constructor
        for more details.
        """
        self.pH = pH
        if weights is None:
            self._weights = [0.0]*len(pH)
            self._updateWeights = True
            self._weightUpdateFactor = 1.0
            self._histogram = [0]*len(pH)
            self._hasMadeTransition = False
        else:
            self._weights = weights
            self._updateWeights = False


    @property
    def weights(self):
        """Get the current values of the weights used in the simulated tempering algorithm.  This has one value for each pH."""
        return [x-self._weights[0] for x in self._weights]

    def attemptMCStep(self):
        """
        Attempt to change the protonation states of all titratable residues.  If simulated tempering is being used, this
        will also attempt to change to a new pH.

        When ``ring_flip_angles`` was passed (and any titratable residue's
        :attr:`TitratableResidueReference.ring_flip_bonds` is non-empty), two
        kinds of terminal-group rotation moves are layered in:

        * One stand-alone ring flip is attempted up-front (random residue,
          random angle with random sign, Metropolis against the implicit
          potential, accepted moves propagated to the explicit context and
          followed by a short solvent relaxation).
        * During the per-residue protonation loop, whenever the residue being
          swapped is a *single-site* proposal and has a ring flip configured,
          50% of the time a flip is additionally proposed *jointly* with the
          protonation state change. The rotation is applied to implicit
          positions before ``newEnergy`` is measured, so the Metropolis
          decision scores the coupled (flip + state-change) move. Accepted
          coupled flips are replayed onto the explicit positions just before
          the trailing solvent relaxation block. Coupled flips are
          deliberately skipped on multisite proposals because rotating the
          primary residue's hydrogens can alter ``_findNeighbors``'s
          membership of the neighbour set, which would break detailed balance
          on the joint move without a Hastings correction.

        The Metropolis acceptance temperature is :attr:`self.temperature`,
        which is read once from the production integrator at construction
        time.
        """
        # Try one ring-flip MC move first. It seeds the implicit context from
        # the explicit context internally and, on acceptance, updates both
        # contexts (plus runs a short solvent relaxation) so the subsequent
        # protonation MC sees a consistent geometry.

        self._attemptRingFlip()

        # Copy the (possibly post-flip) positions to the implicit context.

        state = self.simulation.context.getState(positions=True, parameters=True)
        explicitPositions = state.getPositions(asNumpy=True).value_in_unit(nanometers)
        implicitPositions = explicitPositions[self.implicitAtomIndex]
        self.implicitContext.setPositions(implicitPositions)
        periodicDistance = compiled.periodicDistance(state.getPeriodicBoxVectors().value_in_unit(nanometers))

        # Perform simulated tempering.

        if len(self.pH) > 1:
            self._attemptPHChange()

        # Process the residues in random order.

        anyChange = False
        # Accepted ring flips that were coupled to a protonation swap. The
        # implicit-context positions are already rotated (and were used in the
        # joint-move Metropolis acceptance); we replay the same rotation onto
        # the explicit positions just before the solvent relaxation block.
        # Each entry is ``(residue_index, record_idx, angle_deg)`` so the
        # explicit-side replay can re-locate the same bond record.
        accepted_coupled_flips: list[tuple[int, int, float]] = []
        for resIndex in np.random.permutation(list(self.titrations)):
            titrations = [self.titrations[resIndex]]

            # Single-variant titrations (e.g. ASN, GLN registered purely
            # for the MolProbity-style amide ring flip) carry no
            # protonation transition: there is nothing to swap to and
            # the per-step proposal/Metropolis machinery would just
            # waste an implicit-context energy evaluation plus a
            # solvent-relaxation block on a guaranteed accept.
            # ``_attemptRingFlip`` already attempted the flip move
            # earlier in this method, so just skip the residue here.
            if len(titrations[0].implicitStates) <= 1:
                continue

            # Select a new state for it.

            stateIndex = [self._selectNewState(titrations[0])]
            if np.random.random() < 0.25:
                # Consider a multisite titration in which two residues change.

                neighbors = self._findNeighbors(resIndex, explicitPositions, periodicDistance)
                if len(neighbors) > 0:
                    i = np.random.choice(neighbors)
                    titrations.append(self.titrations[i])
                    stateIndex.append(self._selectNewState(titrations[-1]))

            # Track per-residue MC stats: each residue touched by this
            # proposal (primary, plus any multisite neighbour) gets +1 attempt.
            for t in titrations:
                t.n_state_attempts += 1

            # Compute the energy of the implicit solvent system in the current and new states.

            currentEnergy = self.implicitContext.getState(energy=True).getPotentialEnergy()

            # 50% of the time, couple a terminal-group flip of the primary
            # residue into this protonation proposal (if one is configured).
            # The flip is applied to implicit positions *before* the
            # state-change parameters and *before* ``newEnergy`` is measured,
            # so the Metropolis decision scores the joint move.
            #
            # Only do this for single-site proposals. ``_findNeighbors`` keys
            # the multisite proposal probability off A's hydrogen positions
            # via the 0.2 nm proximity check, and a coupled rotation can move
            # those hydrogens enough to add/remove neighbours - breaking
            # detailed balance because the forward/reverse proposal densities
            # would then differ. Single-site moves have no such dependency.
            coupled_flip: tuple[int, int, float, np.ndarray] | None = None
            records_for_res = self.ringFlips.get(resIndex)
            if (
                len(titrations) == 1
                and records_for_res
                and np.random.random() < 0.5
            ):
                # Multiple rotatable bonds on the same residue are picked
                # uniformly so a ligand with N flip bonds gets N x the
                # per-step probability of any one of them being tried.
                record_idx = int(np.random.randint(len(records_for_res)))
                record = records_for_res[record_idx]
                angle_deg = (
                    float(np.random.choice(record.angles))
                    * float(np.random.choice([-1.0, 1.0]))
                )
                pre_flip_implicit_pos = (
                    self.implicitContext.getState(positions=True)
                    .getPositions(asNumpy=True).value_in_unit(nanometers)
                )
                post_flip_implicit_pos = self._rotateAroundBond(
                    pre_flip_implicit_pos,
                    anchor_index=record.implicit.bond[0],
                    pivot_index=record.implicit.bond[1],
                    rotatable_indices=record.implicit.rotatable_group,
                    angle_deg=angle_deg,
                )
                self.implicitContext.setPositions(post_flip_implicit_pos)
                self.ringFlipAttempts += 1
                self.titrations[resIndex].n_coupled_flip_attempts += 1
                coupled_flip = (resIndex, record_idx, angle_deg, pre_flip_implicit_pos)

            for i, t in zip(stateIndex, titrations, strict=False):
                self._applyStateToContext(t.implicitStates[i], self.implicitContext, self.implicitExceptionIndex, self.implicitInterResidue14, self.implicit14Scale)
            newEnergy = self.implicitContext.getState(energy=True).getPotentialEnergy()

            # Decide whether to accept the new state.

            kT = (MOLAR_GAS_CONSTANT_R*self.temperature)
            deltaRefEnergy = unitsum([t.referenceEnergies[i] - t.referenceEnergies[t.currentIndex] for i, t in zip(stateIndex, titrations, strict=False)])
            deltaN = unitsum([t.implicitStates[i].numHydrogens - t.implicitStates[t.currentIndex].numHydrogens for i, t in zip(stateIndex, titrations, strict=False)])
            w = (newEnergy-currentEnergy-deltaRefEnergy)/kT + deltaN*np.log(10.0)*self.pH[self.currentPHIndex]
            if w > 0.0 and np.exp(-w) < np.random.random():
                # Restore the previous state.

                for t in titrations:
                    self._applyStateToContext(t.implicitStates[t.currentIndex], self.implicitContext, self.implicitExceptionIndex, self.implicitInterResidue14, self.implicit14Scale)
                # Also revert the coupled ring-flip positions if any.
                if coupled_flip is not None:
                    _, _, _, pre_flip_implicit_pos = coupled_flip
                    self.implicitContext.setPositions(pre_flip_implicit_pos)
                continue

            # TODO: Check that there are accepted changes

            anyChange = True

            # Apply the new state.

            for i, t in zip(stateIndex, titrations, strict=False):
                t.currentIndex = i
                t.n_state_accepted += 1
                self._applyStateToContext(t.explicitStates[i], self.simulation.context, self.explicitExceptionIndex, self.explicitInterResidue14, self.explicit14Scale)
                self._applyStateToContext(t.explicitStates[i], self.relaxationContext, self.explicitExceptionIndex, self.explicitInterResidue14, self.explicit14Scale)

            if coupled_flip is not None:
                flip_res_index, flip_record_idx, flip_angle_deg, _ = coupled_flip
                accepted_coupled_flips.append(
                    (flip_res_index, flip_record_idx, flip_angle_deg)
                )
                self.ringFlipAccepted += 1
                self.titrations[flip_res_index].n_coupled_flip_accepted += 1

        # If anything changed, run some dynamics to let the water relax.

        if anyChange:
            # Apply any accepted coupled ring flips to the explicit positions
            # before relaxation, so that the frozen-solute relaxation block
            # picks up the rotated side chains (the relaxation system has
            # zero-mass solute atoms, so they stay put and water adapts).
            explicit_pos_for_relax = explicitPositions
            if accepted_coupled_flips:
                explicit_pos_for_relax = np.asarray(explicitPositions).copy()
                for flip_res_index, flip_record_idx, flip_angle_deg in accepted_coupled_flips:
                    record = self.ringFlips[flip_res_index][flip_record_idx]
                    explicit_pos_for_relax = self._rotateAroundBond(
                        explicit_pos_for_relax,
                        anchor_index=record.explicit.bond[0],
                        pivot_index=record.explicit.bond[1],
                        rotatable_indices=record.explicit.rotatable_group,
                        angle_deg=flip_angle_deg,
                    )
            self.relaxationContext.setPositions(explicit_pos_for_relax)
            self.relaxationContext.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
            for param in self.relaxationContext.getParameters():
                self.relaxationContext.setParameter(param, state.getParameters()[param])
            self.relaxationContext.getIntegrator().step(self.relaxationSteps)
            relaxedPositions = self.relaxationContext.getState(positions=True).getPositions(asNumpy=True)
            self.simulation.context.setPositions(relaxedPositions)

        return anyChange

    @property
    def ringFlipAcceptanceRate(self):
        """Fraction of ring-flip MC moves accepted (0.0 if none attempted)."""
        if self.ringFlipAttempts == 0:
            return 0.0
        return self.ringFlipAccepted / self.ringFlipAttempts

    def reset_stats(self):
        """Zero every per-residue MC counter.

        Useful when you want to discard equilibration statistics before
        beginning a production block. Also clears the
        :class:`~opensqm.md.water_swap_mc.WaterSwapMC` counters when
        water-swap MC is enabled.
        """
        for titration in self.titrations.values():
            titration.reset_stats()
        self.ringFlipAttempts = 0
        self.ringFlipAccepted = 0



    def summary(self):
        """Return per-residue MC acceptance statistics as a ``pandas.DataFrame``.

        Rows are indexed by ``residue_index`` and include the residue name,
        current variant label, and attempt/accept counts plus acceptance
        rates for each move type:

        - ``state_*``: protonation-state swaps. Multisite proposals
          increment both partners' counters.
        - ``standalone_flip_*``: terminal-group rotations proposed on
          their own at the top of :meth:`attemptMCStep`. Only populated
          for residues that have a ring-flip configured.
        - ``coupled_flip_*``: terminal-group rotations proposed jointly
          with a single-site protonation swap. Only populated for
          residues that have a ring-flip configured and are titratable.

        Acceptance rates default to ``0.0`` when no attempts have been
        made so the column dtype stays numeric.
        """
        residues = list(self.explicitTopology.residues())
        rows = []
        residue_indices = sorted(self.titrations.keys())
        for res_index in residue_indices:
            titration = self.titrations[res_index]
            residue = residues[res_index]
            if titration.currentIndex >= 0:
                current_variant = titration.variant_names[titration.currentIndex]
            else:
                current_variant = None
            rows.append({
                "residue_index": res_index,
                "residue_name": residue.name,
                "current_variant": current_variant,
                "has_ring_flip": res_index in self.ringFlips,
                "state_attempts": titration.n_state_attempts,
                "state_accepted": titration.n_state_accepted,
                "state_acceptance_rate": titration.state_acceptance_rate,
                "standalone_flip_attempts": titration.n_standalone_flip_attempts,
                "standalone_flip_accepted": titration.n_standalone_flip_accepted,
                "standalone_flip_acceptance_rate": titration.standalone_flip_acceptance_rate,
                "coupled_flip_attempts": titration.n_coupled_flip_attempts,
                "coupled_flip_accepted": titration.n_coupled_flip_accepted,
                "coupled_flip_acceptance_rate": titration.coupled_flip_acceptance_rate,
            })
        return pd.DataFrame(rows).set_index("residue_index")

    def _attemptRingFlip(self):
        """
        One Metropolis MC attempt to rotate a randomly-chosen terminal group by
        a randomly-chosen angle, evaluated against the implicit potential at
        :attr:`self.temperature`.

        Returns
        -------
        bool
            True if the move was accepted (positions in both the simulation
            context and the implicit context were updated and a relaxation
            block was run), False otherwise.
        """
        if not self.ringFlips:
            return False

        kT = MOLAR_GAS_CONSTANT_R * self.temperature

        # 1) Randomly choose a torsion. Flatten across residues so that a
        # ligand with N rotatable bonds gets N x the per-step probability
        # of being picked, matching the coupled-flip code path.
        flip_slots = [
            (res_idx, rec)
            for res_idx, records in self.ringFlips.items()
            for rec in records
        ]
        slot_idx = int(np.random.randint(len(flip_slots)))
        res_index, record = flip_slots[slot_idx]

        # 2) Randomly choose an angle (random magnitude * random sign).
        angle_deg = (
            float(np.random.choice(record.angles))
            * float(np.random.choice([-1.0, 1.0]))
        )

        # Seed the implicit context from the current explicit positions so the
        # before/after implicit energies are evaluated on a consistent geometry.
        sim_state = self.simulation.context.getState(positions=True, parameters=True)
        explicit_box = sim_state.getPeriodicBoxVectors()
        explicit_pos = sim_state.getPositions(asNumpy=True).value_in_unit(nanometers)
        implicit_pos_before = explicit_pos[self.implicitAtomIndex]
        self.implicitContext.setPositions(implicit_pos_before)
        energy_before = self.implicitContext.getState(getEnergy=True).getPotentialEnergy()

        # 3) Flip in the implicit solvent simulation.
        implicit_pos_after = self._rotateAroundBond(
            implicit_pos_before,
            anchor_index=record.implicit.bond[0],
            pivot_index=record.implicit.bond[1],
            rotatable_indices=record.implicit.rotatable_group,
            angle_deg=angle_deg,
        )
        self.implicitContext.setPositions(implicit_pos_after)
        energy_after = self.implicitContext.getState(getEnergy=True).getPotentialEnergy()

        # 5) Metropolis acceptance.
        self.ringFlipAttempts += 1
        # ``self.ringFlips`` keys are not strictly guaranteed to be a subset
        # of ``self.titrations`` (a caller could register a flip on a
        # non-titrating residue), so look up the titration defensively when
        # bumping per-residue counters.
        stats_titration = self.titrations.get(res_index)
        if stats_titration is not None:
            stats_titration.n_standalone_flip_attempts += 1
        w = (energy_after - energy_before) / kT
        if w > 0.0 and np.exp(-w) < np.random.random():
            # Reject - restore implicit positions to the pre-flip geometry.
            self.implicitContext.setPositions(implicit_pos_before)
            return False

        # 6) Accepted: apply same rotation to the explicit context, then run a
        # short relaxation block so the surrounding solvent can adapt.
        explicit_pos_after = self._rotateAroundBond(
            explicit_pos,
            anchor_index=record.explicit.bond[0],
            pivot_index=record.explicit.bond[1],
            rotatable_indices=record.explicit.rotatable_group,
            angle_deg=angle_deg,
        )
        self.simulation.context.setPositions(explicit_pos_after)
        self.relaxationContext.setPositions(explicit_pos_after)
        self.relaxationContext.setPeriodicBoxVectors(*explicit_box)
        sim_params = sim_state.getParameters()
        for param in self.relaxationContext.getParameters():
            self.relaxationContext.setParameter(param, sim_params[param])
        self.relaxationContext.getIntegrator().step(self.relaxationSteps)
        relaxed_positions = self.relaxationContext.getState(
            positions=True
        ).getPositions(asNumpy=True)
        self.simulation.context.setPositions(relaxed_positions)
        # Keep the implicit context in sync with the post-relaxation geometry
        # so the subsequent protonation MC starts from a consistent state.
        self.implicitContext.setPositions(
            relaxed_positions.value_in_unit(nanometers)[self.implicitAtomIndex]
        )
        self.ringFlipAccepted += 1
        if stats_titration is not None:
            stats_titration.n_standalone_flip_accepted += 1
        return True

    @staticmethod
    def _rotateAroundBond(positions, anchor_index, pivot_index,
                          rotatable_indices, angle_deg):
        """
        Rigid Rodrigues rotation of ``rotatable_indices`` about the axis
        pointing from ``pivot_index`` to ``anchor_index`` by ``angle_deg``.

        The pivot atom is the fixed point of the rotation; the axis points
        outward toward the anchor (i.e. into the rest of the molecule). This
        matches the convention used by
        :func:`opensqm.md.terminal_ring_mc.find_terminal_group`, which returns
        ``bond=(anchor, pivot)`` with ``pivot`` on the rotatable-group side.

        Returns a new ``ndarray`` in nm; ``positions`` is not mutated.
        """
        positions = np.asarray(positions)
        p_pivot = positions[pivot_index]
        axis = positions[anchor_index] - p_pivot
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm == 0.0:
            return positions.copy()
        axis = axis / axis_norm
        theta = float(np.deg2rad(angle_deg))
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotated = positions.copy()
        for idx in rotatable_indices:
            v = positions[idx] - p_pivot
            v_rot = (
                v * cos_t
                + np.cross(axis, v) * sin_t
                + axis * np.dot(axis, v) * (1.0 - cos_t)
            )
            rotated[idx] = p_pivot + v_rot
        return rotated

    def setResidueState(self, residueIndex, stateIndex, relax=False):
        """
        Set a titratable residue to be in a particular state.

        Parameters
        ----------
        residueIndex: int
            The index of the residue to modify
        stateIndex: int
            The index of the state to put it into
        relax: bool
            If True, the solvent is allowed to relax after changing the state by immobilizing the solute and performing
            a short simulation.
        """
        titration = self.titrations[residueIndex]
        self._applyStateToContext(titration.explicitStates[stateIndex], self.simulation.context, self.explicitExceptionIndex, self.explicitInterResidue14, self.explicit14Scale)
        self._applyStateToContext(titration.explicitStates[stateIndex], self.relaxationContext, self.explicitExceptionIndex, self.explicitInterResidue14, self.explicit14Scale)
        self._applyStateToContext(titration.implicitStates[stateIndex], self.implicitContext, self.implicitExceptionIndex, self.implicitInterResidue14, self.implicit14Scale)
        titration.currentIndex = stateIndex
        if relax:
            self.relaxationContext.setPositions(self.simulation.context.getState(positions=True).getPositions(asNumpy=True))
            self.relaxationContext.getIntegrator().step(self.relaxationSteps)
            self.simulation.context.setPositions(self.relaxationContext.getState(positions=True).getPositions(asNumpy=True))

    @staticmethod
    def _applyStateToContext(
        state,
        context,
        exceptionIndex,
        interResidue14,
        coulomb14Scale,
    ):
        """Apply a ``ResidueState`` to a ``Context``.

        Overwrites per-particle and per-exception parameters in the
        requested forces with the state's values.
        """
        for forceIndex, params in state.particleParameters.items():
            force = context.getSystem().getForce(forceIndex)
            is_nb = isinstance(force, NonbondedForce)
            for atomName, atomParams in params.items():
                atomIndex = state.atomIndices[atomName]
                try:
                    # Custom forces take the parameters as a single tuple.
                    force.setParticleParameters(atomIndex, atomParams)
                except:
                    # Standard forces take them as separate arguments.
                    force.setParticleParameters(atomIndex, *atomParams)
            if is_nb:
                for key, exceptionParams in state.exceptionParameters[forceIndex].items():
                    exc_idx = exceptionIndex[key]
                    p = force.getExceptionParameters(exc_idx)
                    force.setExceptionParameters(
                        exc_idx, p[0], p[1], *exceptionParams,
                    )
                for index in interResidue14[state.residueIndex]:
                    p = force.getExceptionParameters(index)
                    p1, p2 = p[0], p[1]
                    sigma, epsilon = p[3], p[4]
                    q1, _, _ = force.getParticleParameters(p1)
                    q2, _, _ = force.getParticleParameters(p2)
                    new_chargeProd = coulomb14Scale * q1 * q2
                    force.setExceptionParameters(
                        index, p1, p2, new_chargeProd, sigma, epsilon,
                    )
            force.updateParametersInContext(context)

    @staticmethod
    def _findInterResidue14(system, topology):
        """For each residue, record the indices of all 1-4 exceptions that span that residue and another one."""
        indices = defaultdict(list)
        atoms = list(topology.atoms())
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                for i in range(force.getNumExceptions()):
                    p1, p2, chargeProd, _sigma, _epsilon = force.getExceptionParameters(i)
                    atom1 = atoms[p1]
                    atom2 = atoms[p2]
                    if atom1.residue != atom2.residue and chargeProd.value_in_unit(elementary_charge**2) != 0.0:
                        indices[atom1.residue.index].append(i)
                        indices[atom2.residue.index].append(i)
        return indices

    @staticmethod
    def _find14Scale(forcefield):
        """Find the scale factor for 1-4 Coulomb interactions."""
        for generator in forcefield.getGenerators():
            if isinstance(generator, NonbondedGenerator):
                return generator.coulomb14scale
        return 1.0

    @staticmethod
    def _findExceptionIndices(system, topology):
        """Construct a dict whose keys are (residue index, atom 1 name, atom 2 name), and whose values are the indices
        of the corresponding exceptions in the NonbondedForce.  This is needed for mapping exceptions between Topologies
        with different sets of atoms.
        """
        indices = {}
        atoms = list(topology.atoms())
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                for i in range(force.getNumExceptions()):
                    p1, p2, _chargeProd, _sigma, _epsilon = force.getExceptionParameters(i)
                    atom1 = atoms[p1]
                    atom2 = atoms[p2]
                    if atom1.residue == atom2.residue:
                        indices[(atom1.residue.index, atom1.name, atom2.name)] = i
                        indices[(atom1.residue.index, atom2.name, atom1.name)] = i
        return indices

    @staticmethod
    def _findResidueStates(topology, positions, forcefield, variants, ffargs, record_residue_indices=None):
        """Given a ForceField and a list of variants for the variable residues, construct ResidueState objects for them."""
        modeller = Modeller(topology, positions)
        modeller.addHydrogens(forcefield=forcefield, variants=variants)
        system = forcefield.createSystem(modeller.topology, **ffargs)
        atoms = list(modeller.topology.atoms())
        residues = list(modeller.topology.residues())
        states: list[ResidueState] = []
        for residue, variant in zip(residues, variants, strict=False):
            if record_residue_indices is not None:
                if residue.index not in record_residue_indices:
                    continue
            elif variant is None:
                continue
            atomIndices = {atom.name: atom.index for atom in residue.atoms()}
            particleParameters = {}
            exceptionParameters = {}
            for i, force in enumerate(system.getForces()):
                try:
                    particleParameters[i] = {atom.name: force.getParticleParameters(atom.index) for atom in residue.atoms()}
                except:
                    pass
                if isinstance(force, NonbondedForce):
                    exceptionParameters[i] = {}
                    for j in range(force.getNumExceptions()):
                        p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(j)
                        atom1 = atoms[p1]
                        atom2 = atoms[p2]
                        if atom1.residue == residue and atom2.residue == residue:
                            exceptionParameters[i][(residue.index, atom1.name, atom2.name)] = (chargeProd, sigma, epsilon)
            numHydrogens = sum(1 for atom in residue.atoms() if atom.element == element.hydrogen)
            states.append(ResidueState(residue.index, atomIndices, particleParameters, exceptionParameters, numHydrogens))
        return states

    @staticmethod
    def _get_zero_parameters(original_parameters, force):
        """Get the per-particle parameter values that should be used to set an atom's charge to 0."""
        p = list(original_parameters)
        if isinstance(force, NonbondedForce) or isinstance(force, GBSAOBCForce):
            p[0] = 0.0
        else:
            for i in range(force.getNumPerParticleParameters()):
                if force.getPerParticleParameterName(i) == 'charge':
                    p[i] = 0.0
        return tuple(p)

    @staticmethod
    def _selectNewState(titration):
        """Randomly choose a new state for a ResidueTitration.

        ``numStates == 1`` is permitted for ring-flip-only residues
        (e.g. ASN, GLN) that are registered for terminal-group MC but
        carry no protonation transitions; the only valid "new" state
        is the current one. The protonation MC loop in
        :meth:`attemptMCStep` skips these residues outright, so the
        no-op return is purely defensive.
        """
        numStates = len(titration.implicitStates)
        if numStates == 1:
            return titration.currentIndex
        if numStates == 2:
            return 1-titration.currentIndex
        stateIndex = titration.currentIndex
        while stateIndex == titration.currentIndex:
            stateIndex = np.random.randint(numStates)
        return stateIndex

    def _findNeighbors(self, resIndex, explicitPositions, periodicDistance):
        """Find other titratable residues that are very close to a specified residue.  This is used for
        multisite titrations.
        """
        neighbors = []
        titration1 = self.titrations[resIndex]
        for resIndex2 in self.titrations:
            if resIndex2 > resIndex:
                titration2 = self.titrations[resIndex2]
                isNeighbor = False
                for i in titration1.explicitHydrogenIndices:
                    for j in titration2.explicitHydrogenIndices:
                        if periodicDistance(explicitPositions[i], explicitPositions[j]) < 0.2:
                            isNeighbor = True
                if isNeighbor:
                    neighbors.append(resIndex2)
        return neighbors

    def _attemptPHChange(self):
        """Attempt to change to a different pH."""
        # Compute the probability for each pH.  This is done in log space to avoid overflow.

        hydrogens = sum(t.explicitStates[t.currentIndex].numHydrogens for t in self.titrations.values())
        logProbability = [(self._weights[i]-hydrogens*np.log(10.0)*self.pH[i]) for i in range(len(self._weights))]
        maxLogProb = max(logProbability)
        offset = maxLogProb + np.log(sum(np.exp(x-maxLogProb) for x in logProbability))
        probability = [np.exp(x-offset) for x in logProbability]
        r = np.random.random_sample()
        for j in range(len(probability)):
            if r < probability[j]:
                if j != self.currentPHIndex:
                    self._hasMadeTransition = True
                self.currentPHIndex = j
                if self._updateWeights:
                    # Update the weight factors.

                    self._weights[j] -= self._weightUpdateFactor
                    self._histogram[j] += 1
                    minCounts = min(self._histogram)
                    if minCounts > 20 and minCounts >= 0.2*sum(self._histogram)/len(self._histogram):
                        # Reduce the weight update factor and reset the histogram.

                        self._weightUpdateFactor *= 0.5
                        self._histogram = [0]*len(self.pH)
                        self._weights = [x-self._weights[0] for x in self._weights]
                    elif not self._hasMadeTransition and probability[self.currentPHIndex] > 0.99 and self._weightUpdateFactor < 1024.0:
                        # Rapidly increase the weight update factor at the start of the simulation to find
                        # a reasonable starting value.

                        self._weightUpdateFactor *= 2.0
                        self._histogram = [0]*len(self.pH)
                        self._weights = [x-self._weights[0] for x in self._weights]
                return
            r -= probability[j]
