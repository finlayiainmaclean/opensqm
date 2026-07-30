"""Implicit-solvent MMGBSA with a protonation-state funnel.

A lightweight cousin of ``run_mmgbsa`` that stops at the cheap implicit-solvent
filter: it never solvates, equilibrates or runs MD. It picks the best protonation
state of the protein and ligand - which protomer, in which pocket - and writes it
as a protonated protein PDB and a ligand SDF, alongside a scores series in the
same schema as ``run_mmgbsa`` (single-point rather than trajectory-averaged).

1. the protein is protonated with PDBFixer + PROPKA/PDB2PQR at the target pH,
   which assigns pH-dependent titration states (HIS/ASP/GLU/CYS/... and the
   HIS/ASN/GLN flips) and optimises the H-bond network - but only sees the apo
   protein, so a residue near the pocket may sit in whichever state best
   satisfies its other protein neighbours rather than the ligand; by default
   (``flip_near_ligand_residues``) every HIS/ASN/GLN within
   ``flip_cutoff_angstrom`` also gets its alternative state(s) tried below -
   a neutral HIS's opposite ring tautomer (HID/HIE), and (independently, for
   HIS/ASN/GLN alike) its terminal ring/amide group rotated 180 degrees about
   its CB-CG or CG-CD bond, the standard "MolProbity flip" that corrects
   X-ray's O/N ambiguity for ASN/GLN and explores ring orientation for HIS;
1b. before any ligand is added, each flip variant's bare protein is minimised in
   GBn2 with a uniform, stiff restraint (``protein_premin_restraint_k``) on every
   heavy atom - relieving whatever local clash PDBFixer/PROPKA protonation, or
   the flip's rigid 180-degree rotation, introduced while there is no ligand in
   the system yet, so it doesn't show up as clash energy in step 3's cheap
   single-point screen;
2. uniKa enumerates every ligand protomer within a free-energy window of the
   solution-dominant one at the target pH;
3. each (residue-flip combination x ligand-protomer) complex is scored in GBn2
   implicit solvent and the best one is selected: every candidate is first
   ranked by a cheap single-point score (no minimisation), and only those
   within ``singlepoint_window_kcal`` of the best are fully minimised, since
   the combinatorics of step 1's flips can otherwise make minimising every
   candidate the dominant cost.

By default a protomer is ranked by ``intrinsic_free_energy + MMGBSA`` - the
bound-state free energy - so the pocket can flip the ligand's protonation state
when binding pays for the intrinsic cost (e.g. an amidinium against an
aspartate), without the blanket bias toward the most-charged protomer that
ranking by raw interaction energy would introduce. Pass
``select_metric="interaction"`` to rank by the bare MMGBSA interaction energy
instead.

The published complex is the winning protomer's minimised structure, split into
``output_dir/prot.pdb`` (protein, PROPKA-protonated) and ``output_dir/lig.sdf``
(ligand, at the minimised pose). By default the protein's crystallographic waters
and ions are kept as explicit residues in the GBn2 minimisation (rather than
stripped) and written into ``prot.pdb``. The prot-lig complex minimisation (step
3) is restrained so GBn2 relaxes only what should move: each kept water's oxygen
is stiffly pinned at its crystal site (hydrogens stay free to reorient), and
every protein heavy atom is held by a restraint graded by its distance to the
ligand - free within ``distal_min_distance`` so the pocket can relax around it,
ramping up to ``distal_max_restraint_force`` at ``distal_max_distance`` and
beyond so the bulk of the protein converges quickly - while the ligand and
near-pocket side chains relax into place. A near-ligand flip's moved atoms are
themselves within the free zone (they were built within ``flip_cutoff_angstrom``
of the ligand), so the restraint does not fight the flip. Pass
``keep_solvent=False`` for the bare protonated protein.

Inputs (protein, ligand) and the output location may each be a local path or an
``s3://`` URI: inputs are staged into a temp dir, all work happens locally, and
only the published artifacts are copied/uploaded to the output location.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath
from loguru import logger
from openff.toolkit.topology import Molecule  # type: ignore
from openmm import LocalEnergyMinimizer, Vec3, unit
from openmm.app import Modeller
from openmm.app.forcefield import ForceField
from openmm.app.pdbfile import PDBFile
from openmm.app.topology import Topology
from pydantic import BaseModel, ConfigDict
from pydantic_units import OpenMMQuantity
from rdkit import Chem, RDLogger
from unipka import UnipKa

from opensqm.fix import (
    enumerate_residue_flip_variants,
    find_flippable_residues,
    run_pdbfixer,
)
from opensqm.md.platforms import make_context, set_platform
from opensqm.md.prepare import (
    create_integrator,
    create_system,
    get_ligand_forcefield,
    strip_solvent,
)
from opensqm.md.run_mmgbsa import (
    _add_position_restraint,
    _enumerate_protomers,
    _ImplicitScorer,
    _minimize_implicit_restrained,
    _ProtomerCandidate,
    _stringify_residue_ids,
)
from opensqm.rdkit_utils import set_coordinates

RDLogger.DisableLog("rdApp.warning")

# How the funnel ranks protomers: "corrected" is intrinsic + MMGBSA (the
# bound-state free energy); "interaction" is the bare MMGBSA interaction energy.
SelectMetric = Literal["corrected", "interaction"]

# GBn2 implicit-solvent force field, plus the TIP3P water and ion templates so
# crystallographic waters/ions kept from the input can be parametrised and
# minimised in the continuum (rather than stripped). Mirrors the explicit
# protein set (see prepare._PROTEIN_FORCEFIELD_FILES) but with GBn2 replacing PME
# water; the water/ion templates are harmless when no such residues are present.
_IMPLICIT_SOLVENT_FORCEFIELD_FILES = (
    "amber/ff14SB.xml",
    "amber/phosaa10.xml",
    "amber/tip3p_standard.xml",
    "amber/tip3p_HFE_multivalent.xml",
    "implicit/gbn2.xml",
)


class MMGBSAImplicitSettings(BaseModel):
    """Settings for the implicit-solvent MMGBSA protonation-state funnel."""

    model_config = ConfigDict(frozen=True)
    ligand_resname: str = "LIG"
    # Skip PDBFixer + PROPKA/PDB2PQR protein preparation (step 1) and the
    # near-ligand HIS flip (step 1b) - no protein modification at all - and
    # load ``protein`` as already prepared/protonated. For when only the
    # ligand protomer enumeration matters (e.g. ``protein`` is already a
    # previous run's ``prot.pdb``) and re-running PDBFixer/PROPKA would be
    # redundant work - and, for an already-fixed structure, can fail outright
    # on chain gaps PDB2PQR doesn't expect twice. Minimisation and scoring
    # (step 3) still run as normal.
    skip_protein_preparation: bool = False
    # pH at which the protein titration states (PROPKA) and the ligand protomers
    # (uniKa) are assigned - the same experimental pH for both, so the protein and
    # ligand protonation states are mutually consistent. Ignored for the protein
    # when ``skip_protein_preparation``.
    ph: float = 7.0
    # Every ligand protomer whose uniKa solution free energy is within this
    # window of the dominant one is scored; the winner minimises the selection
    # metric below. A generous window lets the pocket flip the protonation state
    # away from the solution-dominant one when binding pays for it.
    protonation_penalty: OpenMMQuantity[unit.kilocalories_per_mole] = (
        3.0 * unit.kilocalories_per_mole
    )
    # "corrected" (default) ranks by intrinsic_free_energy + MMGBSA, the
    # bound-state free energy, so a charged protomer must earn its intrinsic
    # cost through binding. "interaction" ranks by the bare MMGBSA interaction
    # energy, which is biased toward the most-charged protomer.
    select_metric: SelectMetric = "corrected"
    # PROPKA/PDB2PQR's H-bond network optimisation (step 1) and PDBFixer's
    # heavy-atom placement only ever see the apo protein, so a HIS/ASN/GLN
    # near the ligand may sit in whichever state best satisfies its *other*
    # protein neighbours rather than the ligand. With this on, every such
    # residue within ``flip_cutoff_angstrom`` of the ligand also gets its
    # alternative state(s) tried, scored through the same GBn2 funnel as the
    # ligand protomers below (every flip-combination x ligand-protomer pair):
    # a neutral HIS's opposite ring tautomer (HID/HIE; already-charged HIP is
    # left alone, it has no other neutral tautomer to flip to), and
    # (independently, for HIS/ASN/GLN alike) its terminal ring/amide group
    # rotated 180 degrees about its CB-CG or CG-CD bond - the standard
    # "MolProbity flip" (see ``opensqm.md.terminal_ring_mc.RING_FLIP_BONDS``).
    flip_near_ligand_residues: bool = True
    flip_cutoff_angstrom: float = 5.0
    # Every combination of the qualifying residues' flip candidates is scored
    # (2**n); this bounds n so a pocket with many nearby HIS/ASN/GLN can't
    # blow up the funnel - only the candidates closest to the ligand are kept
    # when there are more than this allows (see
    # ``enumerate_residue_flip_variants``).
    max_flip_variants: int = 8
    # The full (residue-flip combination x ligand-protomer) candidate set can
    # grow large (2**n_flip_candidates x n_protomers); minimising every one is
    # the expensive step (a GBn2 minimisation vs. one cheap potential-energy
    # call). Every candidate is first scored at a single point (no
    # minimisation, at its as-built geometry) by ``select_metric``, and only
    # those within this many kcal/mol of the best singlepoint score are fully
    # minimised - the rest are far enough behind that minimisation (which
    # mostly relaxes clashes, not the ranking) is very unlikely to close the
    # gap.
    singlepoint_window_kcal: float = 3.0
    # Keep the crystallographic waters and ions the protein carries (PDBFixer
    # protonates them) as explicit residues in the GBn2 minimisation, so they are
    # written into prot.pdb. When False they are stripped and prot.pdb is the bare
    # protein (the standard MMGBSA receptor).
    keep_solvent: bool = True
    # Force constant pinning each kept crystallographic water oxygen to its input
    # position during minimisation (harmonic). Stiff by default so the waters stay
    # at their crystal sites while the ligand, protein side chains and the free
    # water hydrogens relax around them; the restraint biases only the minimised
    # geometry, never the scored interaction energy. Ignored when keep_solvent is
    # False.
    water_restraint_k: OpenMMQuantity[unit.kilocalories_per_mole / unit.angstroms**2] = (
        100.0 * unit.kilocalories_per_mole / unit.angstroms**2
    )
    # Before any prot-lig complex is built, each flip variant's protein alone (no
    # ligand yet) is minimised in GBn2 with a uniform restraint at this force
    # constant on every heavy atom - backbone, side chains and any kept crystal
    # water/ion alike. Stiff enough that the protein barely moves from its input
    # conformation, but enough to relax whatever local clash PDBFixer/PROPKA
    # protonation, or a residue flip's rigid 180-degree rotation, introduced -
    # before that clash energy dominates the single-point screen below.
    # Deliberately a single stage: with no ligand in the system yet, loosening
    # this to a backbone-only pass would let a pocket-lining side chain collapse
    # into the (empty) binding site, clashing badly once the ligand is added.
    protein_premin_restraint_k: OpenMMQuantity[unit.kilocalories_per_mole / unit.angstroms**2] = (
        100.0 * unit.kilocalories_per_mole / unit.angstroms**2
    )
    # Speeds up the prot-lig complex minimisation below: every protein heavy atom
    # (backbone and side chains alike) is restrained by its distance to the
    # ligand instead of a uniform backbone-only hold - free within
    # ``distal_min_distance`` of any ligand atom (so the pocket can relax around
    # it), ramping linearly up to ``distal_max_restraint_force`` at
    # ``distal_max_distance`` and beyond (so the bulk of the protein, far from
    # the ligand, is held essentially rigid and converges quickly). A near-ligand
    # residue flip is built within ``flip_cutoff_angstrom`` of the ligand, so its
    # moved atoms land inside the free zone here and are not fought by the
    # restraint - it does not "know" the atom flipped, only where it starts.
    distal_min_distance: OpenMMQuantity[unit.nanometer] = 0.6 * unit.nanometer
    distal_max_distance: OpenMMQuantity[unit.nanometer] = 1.0 * unit.nanometer
    distal_max_restraint_force: OpenMMQuantity[unit.kilocalories_per_mole / unit.angstroms**2] = (
        100.0 * unit.kilocalories_per_mole / unit.angstroms**2
    )
    # Stiffly pin the ligand heavy atoms at their input pose during minimisation
    # (only side chains and hydrogens relax around a fixed ligand). On by default,
    # since each protomer's pose already starts from the input (near-crystal) pose;
    # this keeps that pose intact rather than letting GBn2 drift it during
    # minimisation, and also keeps a congeneric series at a *consistent* pose so a
    # geometry-sensitive rescore (e.g. SQM) isn't biased toward whichever ligand
    # drifted least. The interaction energy is still scored on the minimised pose.
    restrain_ligand: bool = True
    ligand_restraint_k: OpenMMQuantity[unit.kilocalories_per_mole / unit.angstroms**2] = (
        4.0 * unit.kilocalories_per_mole / unit.angstroms**2
    )


@dataclass
class MMGBSAImplicitResult:
    """Outputs of one implicit-solvent MMGBSA funnel run.

    ``protein_path``/``ligand_path`` are the published protonated protein PDB and
    winning-protomer ligand SDF (same scheme as the run's ``output``). ``scores``
    is the winner's score series in the same schema as :class:`run_mmgbsa`'s (a
    single minimised structure rather than a trajectory, so ``interaction_energy``,
    ``interaction_energy_min`` coincide, ``interaction_energy_std`` is 0 and
    ``n_frames``/``n_replicas`` are 1). ``protomers`` is the per-protomer table
    (one row each, with a ``selected`` flag).
    """

    protein_path: str
    ligand_path: str
    scores: pd.Series
    protomers: pd.DataFrame


def _build_implicit_complex(
    protomer: _ProtomerCandidate,
    protein_modeller: Modeller,
    ligand_resname: str,
    keep_solvent: bool,
) -> tuple[Topology, unit.Quantity, ForceField]:
    """Build the ligand-protein complex in GBn2 implicit solvent (no box).

    Unlike ``run_mmgbsa``'s implicit build, the protein's crystallographic waters
    and ions are kept by default (``keep_solvent``): the force field loads the
    TIP3P water/ion templates so they are parametrised and minimised in the GBn2
    continuum rather than stripped. The ligand is added first (residues renamed to
    ``ligand_resname``), matching the atom ordering the writers rely on.
    """
    offmol = Molecule.from_rdkit(protomer.mol, allow_undefined_stereo=True)
    lig_top = offmol.to_topology().to_openmm()
    lig_pos = (offmol.conformers[0].m * unit.angstrom).in_units_of(unit.nanometer)
    for chain in lig_top.chains():
        if not str(chain.id).strip():
            chain.id = "L"
        for res in chain.residues():
            res.name = ligand_resname

    forcefield = get_ligand_forcefield([offmol])
    forcefield.loadFile(_IMPLICIT_SOLVENT_FORCEFIELD_FILES)

    protein = (
        protein_modeller
        if keep_solvent
        else strip_solvent(Modeller(protein_modeller.topology, protein_modeller.positions))
    )
    modeller = Modeller(lig_top, lig_pos)
    modeller.add(protein.topology, protein.positions)
    return modeller.topology, modeller.positions, forcefield


def _minimize_protein_only(
    topology: Topology,
    positions: unit.Quantity,
    config: "MMGBSAImplicitSettings",
) -> unit.Quantity:
    """Minimise the bare protein (no ligand yet) in GBn2, pinned by a uniform restraint.

    Run once per flip variant before it is used to build any prot-lig complex, so
    whatever local clash PDBFixer/PROPKA protonation - or a residue flip's rigid
    180-degree rotation - introduced is relieved while it's cheap (no ligand atoms
    in the system yet), rather than showing up as clash energy in the single-point
    screen below. ``protein_premin_restraint_k`` restrains every heavy atom -
    protein, kept crystal water and ions alike - stiffly enough that the protein
    barely moves from its input conformation. Deliberately does not loosen to a
    backbone-only pass afterward: with no ligand in the system yet, nothing keeps
    the binding pocket open, so a fully side-chain-free relaxation here lets a
    pocket-lining residue collapse into the (empty) binding site, clashing badly
    once the ligand is added at its native pose.
    """
    forcefield = ForceField(*_IMPLICIT_SOLVENT_FORCEFIELD_FILES)
    system = create_system(forcefield, topology, implicit_solvent=True)
    coords_nm = np.asarray(positions.value_in_unit(unit.nanometer))
    # Every heavy atom in the topology - protein backbone/side chains, any kept
    # crystal water's oxygen, and any ion (itself a single heavy atom) - leaving
    # only hydrogens (protein and water alike) free to reorient.
    _add_position_restraint(
        system,
        coords_nm,
        [a.index for a in topology.atoms() if a.element is not None and a.element.symbol != "H"],
        config.protein_premin_restraint_k,
    )
    context = make_context(system, create_integrator(0.002 * unit.picoseconds))
    context.setPositions(positions)
    LocalEnergyMinimizer.minimize(context, maxIterations=1000)
    return context.getState(getPositions=True).getPositions()


def _minimize(
    forcefield: ForceField,
    topology: Topology,
    positions: unit.Quantity,
    config: "MMGBSAImplicitSettings",
) -> np.ndarray:
    """Minimise the implicit complex with the shared water + distal-protein restraints.

    Delegates to ``run_mmgbsa._minimize_implicit_restrained`` so this tool and
    ``run_mmgbsa`` share the same minimisation machinery: ``keep_solvent`` toggles
    the water-oxygen pin (there are no waters to pin once solvent is stripped) and
    the protein heavy atoms are held by the distance-graded restraint (free near
    the ligand, stiff far from it) rather than a uniform backbone-only hold.
    Returns the minimised coordinates (n_atoms x 3, nm).
    """
    return _minimize_implicit_restrained(
        forcefield,
        topology,
        positions,
        restrain_water=config.keep_solvent,
        water_restraint_k=config.water_restraint_k,
        restrain_backbone=False,
        restrain_ligand=config.restrain_ligand,
        ligand_restraint_k=config.ligand_restraint_k,
        restrain_protein_distal=True,
        distal_min_distance=config.distal_min_distance,
        distal_max_distance=config.distal_max_distance,
        distal_max_restraint_force=config.distal_max_restraint_force,
        ligand_resname=config.ligand_resname,
    )


def _write_outputs(
    topology: Topology,
    positions_nm: np.ndarray,
    ligand_mol: Chem.Mol,
    ligand_resname: str,
    prot_path: Path,
    lig_path: Path,
) -> None:
    """Split a minimised implicit complex into a protein PDB and a ligand SDF.

    ``positions_nm`` are the full complex's minimised coordinates (n_atoms x 3,
    nm). The implicit complex places the ligand first and preserves the RDKit atom
    order of ``ligand_mol`` (OpenFF -> OpenMM keep atom order), so the ligand's
    frame coordinates map onto the RDKit mol positionally.
    """
    _stringify_residue_ids(topology)
    positions = unit.Quantity([Vec3(*row) for row in positions_nm], unit.nanometer)

    prot_modeller = Modeller(topology, positions)
    prot_modeller.delete(
        [a for a in prot_modeller.topology.atoms() if a.residue.name == ligand_resname]
    )
    PDBFile.writeFile(
        prot_modeller.topology, prot_modeller.positions, prot_path.open("w"), keepIds=True
    )

    lig_idx = [a.index for a in topology.atoms() if a.residue.name == ligand_resname]
    lig_coords_ang = positions_nm[lig_idx] * 10.0
    ligand_out = set_coordinates(ligand_mol, coords=lig_coords_ang)
    Chem.MolToMolFile(ligand_out, str(lig_path))


def run_mmgbsa_implicit(
    protein: str,
    ligand: str,
    output: str,
    config: MMGBSAImplicitSettings | None = None,
) -> MMGBSAImplicitResult:
    """Implicit-solvent MMGBSA that picks the best protonation state.

    PDBFixer + PROPKA protonate the protein and try the near-ligand HIS/ASN/GLN
    flips (unless ``config.skip_protein_preparation``, which skips both and trusts
    ``protein`` as already prepared/protonated) and uniKa enumerates the ligand's
    protomers within
    ``config.protonation_penalty`` of the solution-dominant one at the target pH. Each
    protomer's complex is minimised and scored in GBn2
    implicit solvent, and the one minimising ``config.select_metric`` is written
    out as the protonated protein (``prot.pdb``) and the winning-protomer ligand
    (``lig.sdf``), both at the minimised pose. With ``config.keep_solvent`` (the
    default) the protein's crystallographic waters and ions are minimised with the
    complex and written into ``prot.pdb``.

    ``protein``, ``ligand`` and ``output`` may each be a local path or an
    ``s3://`` URI. Returns a :class:`MMGBSAImplicitResult` with the published
    locations, a ``scores`` series in ``run_mmgbsa``'s schema, and the per-protomer
    table.
    """
    if config is None:
        config = MMGBSAImplicitSettings()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)

        # Stage inputs locally (downloading from S3 when needed). All heavy work
        # runs in the temp dir; only the published artifacts below are copied out.
        protein_src, ligand_src = AnyPath(protein), AnyPath(ligand)
        local_protein = tmp_dir / f"protein_input{protein_src.suffix or '.pdb'}"
        local_protein.write_bytes(protein_src.read_bytes())
        local_ligand = tmp_dir / f"ligand_input{ligand_src.suffix or '.sdf'}"
        local_ligand.write_bytes(ligand_src.read_bytes())

        fixed_protein = tmp_dir / "protein_prepared.pdb"
        prot_path = tmp_dir / "prot.pdb"
        lig_path = tmp_dir / "lig.sdf"
        score_path = tmp_dir / "scores.csv"
        protomers_path = tmp_dir / "protomers.csv"

        # 1. Protonate the protein (PROPKA titration states, shared by every
        #    protomer) - unless ``skip_protein_preparation``, which trusts
        #    ``protein`` as already prepared/protonated and loads it as-is.
        if config.skip_protein_preparation:
            protein_pdb = PDBFile(str(local_protein))
        else:
            run_pdbfixer(local_protein, fixed_protein, ph=config.ph)
            protein_pdb = PDBFile(str(fixed_protein))
        protein_modeller = Modeller(protein_pdb.topology, protein_pdb.positions)

        # 1b. Enumerate near-ligand HIS/ASN/GLN flips (tautomer for HIS, ring/amide
        #     180-degree rotation for HIS/ASN/GLN alike). PROPKA/PDB2PQR (step 1) and
        #     PDBFixer only ever see the apo protein, so such a residue near the
        #     pocket may sit in whichever state best satisfies its other protein
        #     neighbours rather than the ligand; each combination is scored below
        #     alongside the ligand protomers so the pocket can pick instead. Always
        #     at least the unflipped baseline ("" label = PDB2PQR's own assignment,
        #     unchanged). Skipped entirely under ``skip_protein_preparation`` - that
        #     mode makes no protein modifications at all, trusting the input as
        #     already prepared.
        flip_variants: list[tuple[str, Topology, unit.Quantity]] = [
            ("", protein_modeller.topology, protein_modeller.positions)
        ]
        if config.flip_near_ligand_residues and not config.skip_protein_preparation:
            ligand_ref_mol = Chem.MolFromMolFile(str(local_ligand), removeHs=False)
            ligand_ref_coords = np.array(ligand_ref_mol.GetConformer().GetPositions())
            flippable = find_flippable_residues(
                protein_modeller.topology,
                protein_modeller.positions,
                ligand_ref_coords,
                cutoff_angstrom=config.flip_cutoff_angstrom,
            )
            if flippable:
                logger.info(
                    f"{len(flippable)} HIS/ASN/GLN flip candidate(s) within "
                    f"{config.flip_cutoff_angstrom} A of the ligand; trying alternate "
                    "states alongside PDB2PQR's own assignment"
                )
                flip_variants = enumerate_residue_flip_variants(
                    protein_modeller.topology,
                    protein_modeller.positions,
                    flippable,
                    max_variants=config.max_flip_variants,
                )

        # 2. Enumerate ligand protomers within the penalty window at the target pH.
        penalty_kcal = config.protonation_penalty.value_in_unit(unit.kilocalories_per_mole)
        protomers = _enumerate_protomers(local_ligand, config.ph, penalty_kcal, UnipKa())

        # 3a. Build every (residue-flip combo x ligand protomer) candidate's implicit
        #     complex and scorer. Building is cheap relative to minimising (one
        #     force-field parametrisation vs. a GBn2 minimisation), so this always
        #     runs in full; it's minimisation below that gets filtered down.
        metric_key = "corrected_kcal" if config.select_metric == "corrected" else "interaction_kcal"
        candidates: list[
            tuple[str, _ProtomerCandidate, Topology, unit.Quantity, ForceField, _ImplicitScorer]
        ] = []
        for flip_label, flip_topology, flip_positions in flip_variants:
            logger.info(
                f"Pre-minimising protein alone (flip state: {flip_label or 'PDB2PQR default'})"
            )
            preminimized_positions = _minimize_protein_only(flip_topology, flip_positions, config)
            flip_modeller = Modeller(flip_topology, preminimized_positions)
            for protomer in protomers:
                logger.info(
                    f"Building implicit complex for {protomer.smiles} "
                    f"(flip state: {flip_label or 'PDB2PQR default'})"
                )
                topology, positions, forcefield = _build_implicit_complex(
                    protomer, flip_modeller, config.ligand_resname, config.keep_solvent
                )
                scorer = _ImplicitScorer(topology, positions, forcefield, config.ligand_resname)
                candidates.append((flip_label, protomer, topology, positions, forcefield, scorer))

        # 3b. Score every candidate at a single point (its as-built geometry, no
        #     minimisation) and keep only those within ``singlepoint_window_kcal``
        #     of the best by ``select_metric`` for the expensive full minimisation
        #     below. Skipped for a single candidate - there is nothing to filter.
        if len(candidates) > 1:

            def _singlepoint_score(
                candidate: tuple[
                    str, _ProtomerCandidate, Topology, unit.Quantity, ForceField, _ImplicitScorer
                ],
            ) -> float:
                flip_label, protomer, _topology, positions, _forcefield, scorer = candidate
                pos_nm = np.array(positions.value_in_unit(unit.nanometer))
                interaction = scorer.interaction_energy(pos_nm)
                if metric_key == "corrected_kcal":
                    score = protomer.intrinsic_kcal + interaction
                else:
                    score = interaction
                logger.info(
                    f"[singlepoint] {protomer.smiles} (flip {flip_label or 'default'}): "
                    f"interaction {interaction:.2f}, score {score:.2f} kcal/mol"
                )
                return score

            scored = [(candidate, _singlepoint_score(candidate)) for candidate in candidates]
            best_score = min(score for _candidate, score in scored)
            window_kcal = config.singlepoint_window_kcal
            candidates = [
                candidate for candidate, score in scored if score - best_score <= window_kcal
            ]
            logger.info(
                f"Single-point screen: {len(scored)} candidates -> fully minimising "
                f"the {len(candidates)} within {window_kcal} kcal/mol of the best "
                f"by {config.select_metric}"
            )

        # 3c. Fully minimise + score the surviving candidates, keeping each minimised
        #     complex so the winner's can be written out below.
        records: list[dict] = []
        # Per candidate, aligned with ``records``: the scored protomer and the
        # (topology, pre-min positions, minimised coords nm) of its built complex.
        combo_protomers: list[_ProtomerCandidate] = []
        built: list[tuple[Topology, unit.Quantity, np.ndarray]] = []
        for flip_label, protomer, topology, positions, forcefield, scorer in candidates:
            logger.info(f"Minimising complex for {protomer.smiles}")
            minimised = _minimize(forcefield, topology, positions, config)
            interaction = scorer.interaction_energy(minimised)
            corrected = protomer.intrinsic_kcal + interaction
            records.append(
                {
                    "smiles": protomer.smiles,
                    "charge": protomer.charge,
                    "intrinsic_kcal": protomer.intrinsic_kcal,
                    "flip_state": flip_label,
                    "interaction_kcal": interaction,
                    "corrected_kcal": corrected,
                    "selected": False,
                }
            )
            combo_protomers.append(protomer)
            built.append((topology, positions, minimised))
            logger.info(
                f"[implicit-min] {protomer.smiles} (charge {protomer.charge:+d}, "
                f"flip {flip_label or 'default'}): interaction {interaction:.2f} + "
                f"intrinsic {protomer.intrinsic_kcal:.2f} = corrected {corrected:.2f} kcal/mol"
            )

        # 4. Select the winner on the chosen metric.
        winner_idx = min(range(len(records)), key=lambda i: records[i][metric_key])
        records[winner_idx]["selected"] = True
        winner = combo_protomers[winner_idx]
        winner_record = records[winner_idx]
        logger.info(
            f"Selected {winner.smiles} (charge {winner.charge:+d}) by {config.select_metric}: "
            f"interaction {winner_record['interaction_kcal']:.2f}, "
            f"corrected {winner_record['corrected_kcal']:.2f} kcal/mol, "
            f"flip state: {winner_record['flip_state'] or 'PDB2PQR default'}"
        )

        protomers_df = pd.DataFrame(records)
        protomers_df.to_csv(protomers_path, index=False)
        if len(records) > 1:
            logger.info(f"Protomer funnel:\n{protomers_df.to_string(index=False)}")

        # 5. Write the winner's minimised complex, split into protein and ligand.
        topology, winner_pre, minimised = built[winner_idx]
        _write_outputs(topology, minimised, winner.mol, config.ligand_resname, prot_path, lig_path)

        # Ligand heavy-atom RMSD from the input pose to the minimised pose (A). With
        # the backbone restrained the complex frame is stable, so this is the
        # single-point analogue of run_mmgbsa's MD ligand-drift metric: how far the
        # pose relaxed during minimisation.
        pre_ang = np.asarray(winner_pre.value_in_unit(unit.angstrom))
        post_ang = minimised * 10.0
        lig_heavy = [
            a.index
            for a in topology.atoms()
            if a.residue.name == config.ligand_resname
            and a.element is not None
            and a.element.symbol != "H"
        ]
        ligand_rmsd = (
            float(np.sqrt(((pre_ang[lig_heavy] - post_ang[lig_heavy]) ** 2).sum(axis=1).mean()))
            if lig_heavy
            else 0.0
        )

        # 7. Score series in run_mmgbsa's schema. A single minimised structure, so
        #    the mean/min interaction coincide, the std is 0 and n_frames/n_replicas
        #    are 1; the meaning of each key otherwise matches run_mmgbsa.

        interaction = float(winner_record["interaction_kcal"])
        score_data: dict[str, float | str | int] = {
            "interaction_energy": interaction,
            "interaction_energy_std": 0.0,
            "interaction_energy_min": interaction,
            "n_frames": 1,
            "n_replicas": 1,
            "ligand_rmsd": ligand_rmsd,
            "ligand_charge": int(winner.charge),
            "protonation_penalty": float(winner.intrinsic_kcal),
            "mmgbsa_score": float(winner_record["corrected_kcal"]),
            "flip_state": str(winner_record["flip_state"]),
        }
        scores = pd.Series(score_data)
        scores.to_csv(score_path, header=False)
        logger.info(
            f"MMGBSA (implicit, minimised) = {interaction:.2f} kcal/mol, "
            f"score {scores['mmgbsa_score']:.2f} (ligand RMSD {ligand_rmsd:.2f} A)."
        )

        # Publish the artifacts to the destination (local dir or S3 prefix).
        out_dir = AnyPath(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_prot, out_lig, out_scores = (
            out_dir / "prot.pdb",
            out_dir / "lig.sdf",
            out_dir / "scores.csv",
        )
        out_prot.write_bytes(prot_path.read_bytes())
        out_lig.write_bytes(lig_path.read_bytes())
        out_scores.write_bytes(score_path.read_bytes())
        (out_dir / "protomers.csv").write_bytes(protomers_path.read_bytes())

        logger.info(f"Saved scores to {out_scores}")
        logger.info(f"Saved protonated protein to {out_prot}")
        logger.info(f"Saved best ligand protomer to {out_lig}")

    return MMGBSAImplicitResult(
        protein_path=str(out_prot),
        ligand_path=str(out_lig),
        scores=scores,
        protomers=protomers_df,
    )


@click.command()
@click.option("--protein", required=True, help="Protein PDB file (local path or s3:// URI).")
@click.option("--ligand", required=True, help="Ligand MOL/SDF file (local path or s3:// URI).")
@click.option("--output", required=True, help="Output directory (local path or s3:// prefix).")
@click.option(
    "--ph",
    default=7.0,
    show_default=True,
    help="pH for both the protein PROPKA titration states and the ligand protomers.",
)
@click.option(
    "--skip-protein-preparation/--prepare-protein",
    default=False,
    show_default=True,
    help="Load --protein as already prepared/protonated, skipping PDBFixer/PROPKA "
    "and the near-ligand HIS flip (no protein modification at all). Minimisation "
    "and scoring still run as normal. For re-running just the ligand protomer "
    "funnel against a previous run's prot.pdb.",
)
@click.option(
    "--platform",
    "platform",
    type=click.Choice(["cuda", "mps", "cpu"], case_sensitive=False),
    default=None,
    help="Force the OpenMM compute platform: 'cuda' (NVIDIA GPU), 'mps' "
    "(Apple Silicon Metal/OpenCL), or 'cpu'. Fails if unavailable. "
    "Default: OpenMM auto-selects the fastest platform.",
)
def main(
    protein: str,
    ligand: str,
    output: str,
    ph: float,
    skip_protein_preparation: bool,
    platform: str | None,
) -> None:
    """Run an implicit-solvent MMGBSA protonation funnel from the command line."""
    set_platform(platform)
    config = MMGBSAImplicitSettings(
        ph=ph,
        protonation_penalty=3 * unit.kilocalories_per_mole,
        keep_solvent=True,
        skip_protein_preparation=skip_protein_preparation,
    )
    result = run_mmgbsa_implicit(protein, ligand, output, config=config)
    logger.info(f"Scores:\n{result.scores}")
    logger.info(f"Protein: {result.protein_path}")
    logger.info(f"Ligand: {result.ligand_path}")


if __name__ == "__main__":
    main()
