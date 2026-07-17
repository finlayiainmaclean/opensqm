"""MMGBSA interaction energy with a protonation-state funnel.

The protein is protonated with PDBFixer. Rather than trust uniKa's single
solution-dominant ligand protomer (which ignores the pocket - e.g. it calls
benzamidine neutral, though it binds trypsin as the cationic amidinium), every
protomer within a free-energy window of the dominant one at the target pH is
funnelled from cheap to expensive:

1. each protomer is minimised and scored in implicit solvent;
2. the top-k advance to a short implicit-solvent MD, scored over its frames;
3. the single best of those earns a full explicit run - solvation, equilibration
   (NVT warmup + NPT) and production - scored by the n-closest-waters MMGBSA.

At each stage a protomer is ranked by ``intrinsic_free_energy + MMGBSA`` - the
bound-state free energy - so the pocket can flip the protonation state when
binding pays for the intrinsic cost, without a blanket bias toward the
most-charged protomer. The published score and the representative snapshot
(lowest-energy frame, split into a protein plus-closest-waters PDB and a
protonated-ligand SDF) come from the winner's full explicit run.

Inputs (protein, ligand) and the output location may each be a local path or an
``s3://`` URI: inputs are staged into a temp dir, all work happens locally, and
only the published artifacts are copied/uploaded to the output location.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import click
import mdtraj as md
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath
from loguru import logger
from openmm import Context, LocalEnergyMinimizer, Vec3, unit
from openmm.app import Modeller
from openmm.app.forcefield import ForceField
from openmm.app.pdbfile import PDBFile
from openmm.app.topology import Topology
from pydantic import BaseModel, ConfigDict, Field
from pydantic_units import OpenMMQuantity
from rdkit import Chem, RDLogger
from scipy.spatial.distance import cdist
from unipka import UnipKa

from opensqm.cph.reference_energy import build_protonation_states
from opensqm.cph.run_cph import SystemState
from opensqm.fix import run_pdbfixer
from opensqm.md.equilibrate import (
    EquilibrationSettings,
    _recenter_ligand_positions,
    equilibrate,
)
from opensqm.md.mmgbsa import get_interaction_energy
from opensqm.md.prepare import create_integrator, create_system, prepare_complex
from opensqm.md.vanilla import ProductionSettings, production
from opensqm.rdkit_utils import set_coordinates, set_residue_info

RDLogger.DisableLog("rdApp.warning")

_WATER_RESNAMES = ("HOH", "WAT", "TIP3", "TIP4", "TIP5")
_ION_RESNAMES = ("NA", "CL")


def _stringify_residue_ids(topology: Topology) -> None:
    """Coerce residue ids to strings so ``PDBFile.writeFile(keepIds=True)`` works.

    ``prepare_*``/``addSolvent`` and ``mdtraj``'s ``to_openmm`` can leave residue
    ids as ints, which ``keepIds=True`` rejects (it length-checks the id). See the
    same coercion in ``run_cph``.
    """
    for res in topology.residues():
        res.id = str(res.id)


class MMGBSASettings(BaseModel):
    """Settings for the MMGBSA calculation."""

    model_config = ConfigDict(frozen=True)
    n_closest_waters: int = 5
    ligand_resname: str = "LIG"
    # pH at which the ligand's protomers are enumerated (uniKa).
    protomer_ph: float = 7.0
    # Every ligand protomer whose uniKa solution free energy is within this
    # window of the dominant one is scored; the winner minimises
    # (this intrinsic free energy + MMGBSA). This lets the binding pocket flip
    # the protonation state away from the solution-dominant one when binding
    # pays for it (e.g. an amidinium against an aspartate), without a blanket
    # bias toward the most-charged protomer.
    protonation_penalty: OpenMMQuantity[unit.kilocalories_per_mole] = (
        3.0 * unit.kilocalories_per_mole
    )
    # Production MD time for the winning protomer's full explicit run. MMGBSA is
    # scored over this trajectory and its lowest-energy frame becomes the
    # representative pose.
    production_time: OpenMMQuantity[unit.nanosecond] = 0.5 * unit.nanosecond
    # Independent production replicas for the winning protomer's full explicit
    # run. Each replica reseeds velocities from the shared equilibrated state, so
    # they sample independently; their per-frame energies are pooled for the score
    # and the global lowest-energy frame becomes the representative pose.
    n_replicas: int = 3
    # Protomers are funnelled cheaply before the winner earns the full explicit
    # run: every enumerated protomer is minimised and scored in implicit solvent,
    # and the single best of those is run explicitly.
    equilibration_config: EquilibrationSettings = Field(default_factory=EquilibrationSettings)


@dataclass(frozen=True)
class _ProtomerCandidate:
    """One ligand protomer to score, with its uniKa solution free-energy cost.

    ``mol`` is an explicit-H, conformer-bearing RDKit mol in the input binding
    pose. ``intrinsic_kcal`` is the uniKa pH-adjusted free energy relative to the
    solution-dominant protomer (0 for the dominant one), i.e. the free-energy
    penalty for choosing this protonation state before any binding is considered.
    """

    mol: Chem.Mol
    smiles: str
    charge: int
    intrinsic_kcal: float


def _enumerate_protomers(
    ligand_path: Path, ph: float, penalty_kcal: float, unipka: UnipKa
) -> list[_ProtomerCandidate]:
    """Enumerate ligand protomers within ``penalty_kcal`` of the dominant one at ``ph``.

    ``get_distribution`` transplants the input pose onto every protomer; each one
    kept is realised as an explicit-H, conformer-bearing mol via
    ``build_protonation_states`` (so newly (de)protonated sites get sensible
    geometry). Its own transplanted conformer is the geometry reference - the
    input pose, but at the protomer's protonation, so the constrained embed
    matches rather than fighting a differently-protonated input core. Ordered by
    increasing intrinsic free energy (dominant first).
    """
    molecule_rdmol = set_residue_info(Chem.MolFromMolFile(str(ligand_path), removeHs=False))
    distribution = unipka.get_distribution(molecule_rdmol, pH=ph).reset_index(drop=True)
    within_window = distribution[
        distribution["relative_ph_adjusted_free_energy"] < penalty_kcal
    ].sort_values("relative_ph_adjusted_free_energy")

    candidates: list[_ProtomerCandidate] = []
    for _, row in within_window.iterrows():
        smiles = str(row["smiles"])
        intrinsic = float(row["relative_ph_adjusted_free_energy"])
        try:
            [mol] = build_protonation_states([row["mol"]])
        except Exception as exc:  # a rare protomer may fail to embed; skip it
            logger.warning(f"Skipping protomer {smiles} (charge {row['charge']}): {exc}")
            continue
        candidates.append(
            _ProtomerCandidate(
                mol=mol,
                smiles=smiles,
                charge=int(row["charge"]),
                intrinsic_kcal=intrinsic,
            )
        )

    if not candidates:
        raise RuntimeError("uniKa enumerated no usable ligand protomer within the penalty window")
    logger.info(
        f"Scoring {len(candidates)} ligand protomer(s) within {penalty_kcal} kcal/mol at pH {ph}: "
        + ", ".join(f"{c.smiles} (q{c.charge:+d}, +{c.intrinsic_kcal:.2f})" for c in candidates)
    )
    return candidates


class _ImplicitScorer:
    """Fast implicit-solvent MMGBSA scorer for one protomer's ligand-protein complex.

    Builds the complex, protein-only and ligand-only implicit-solvent (GBn2)
    systems once and reuses their contexts, so the interaction energy
    ``E_complex - E_protein - E_ligand`` can be evaluated cheaply for a minimised
    structure or every frame of a short implicit MD. This is the funnel's coarse
    filter - no explicit waters, no box, no equilibration - used only to rank
    protomers before the winner earns a full explicit run.
    """

    def __init__(
        self,
        topology: Topology,
        positions: unit.Quantity,
        forcefield: ForceField,
        ligand_resname: str = "LIG",
    ) -> None:
        self.topology = topology
        self.lig_idx = np.array(
            [a.index for a in topology.atoms() if a.residue.name == ligand_resname], dtype=np.int64
        )
        self.prot_idx = np.array(
            [a.index for a in topology.atoms() if a.residue.name != ligand_resname], dtype=np.int64
        )

        prot_mod = Modeller(topology, positions)
        prot_mod.delete([a for a in prot_mod.topology.atoms() if a.residue.name == ligand_resname])
        lig_mod = Modeller(topology, positions)
        lig_mod.delete([a for a in lig_mod.topology.atoms() if a.residue.name != ligand_resname])

        # The complex forcefield (ligand SMIRNOFF + amber + GBn2) parametrises the
        # protein-only and ligand-only sub-topologies too.
        self._complex_ctx = Context(
            create_system(forcefield, topology, implicit_solvent=True),
            create_integrator(0.002 * unit.picoseconds),
        )
        self._prot_ctx = Context(
            create_system(forcefield, prot_mod.topology, implicit_solvent=True),
            create_integrator(0.002 * unit.picoseconds),
        )
        self._lig_ctx = Context(
            create_system(forcefield, lig_mod.topology, implicit_solvent=True),
            create_integrator(0.002 * unit.picoseconds),
        )

    @staticmethod
    def _potential(ctx: Context, positions_nm: np.ndarray) -> float:
        ctx.setPositions(positions_nm * unit.nanometer)
        return (
            ctx.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilocalories_per_mole)
        )

    def interaction_energy(self, positions_nm: np.ndarray) -> float:
        """MMGBSA interaction energy (kcal/mol) at ``positions_nm`` (n_atoms x 3, nm)."""
        e_complex = self._potential(self._complex_ctx, positions_nm)
        e_protein = self._potential(self._prot_ctx, positions_nm[self.prot_idx])
        e_ligand = self._potential(self._lig_ctx, positions_nm[self.lig_idx])
        return e_complex - e_protein - e_ligand

    def minimize(self, positions: unit.Quantity) -> np.ndarray:
        """Minimise the complex in implicit solvent; return the minimised coords (nm)."""
        self._complex_ctx.setPositions(positions)
        LocalEnergyMinimizer.minimize(self._complex_ctx, maxIterations=1000)
        return np.asarray(
            self._complex_ctx.getState(getPositions=True)
            .getPositions()
            .value_in_unit(unit.nanometer)
        )


def _build_implicit_complex(
    protomer: _ProtomerCandidate, protein_modeller: Modeller
) -> tuple[Topology, unit.Quantity, ForceField]:
    """Build the ligand-protein complex in implicit solvent (waters stripped, no box)."""
    return prepare_complex(
        protomer.mol,
        protein_modeller=protein_modeller,
        solvent_mode="implicit",
    )


@dataclass
class _PreparedComplex:
    """A protomer's solvated, equilibrated complex, ready to run production from.

    Holds the equilibrated OpenMM ``topology``/``positions``/``forcefield`` so
    production MD can be launched at any length (a short screen and, for the
    winner, the full run) without re-equilibrating.
    """

    protomer: _ProtomerCandidate
    topology: Topology
    positions: unit.Quantity
    forcefield: ForceField
    equilibrated_pdb: Path
    protomer_sdf: Path
    work_dir: Path


def _prepare_and_equilibrate(
    protomer: _ProtomerCandidate,
    protein_modeller: Modeller,
    config: MMGBSASettings,
    work_dir: Path,
) -> _PreparedComplex:
    """Build, solvate and equilibrate (NVT warmup + NPT) one protomer's complex."""
    work_dir.mkdir(parents=True, exist_ok=True)
    protomer_sdf = work_dir / "protomer.sdf"
    Chem.MolToMolFile(protomer.mol, str(protomer_sdf))

    topology, positions, forcefield = prepare_complex(
        protomer.mol,
        protein_modeller=protein_modeller,
        solvent_mode="explicit",
    )
    _stringify_residue_ids(topology)

    logger.info(f"Equilibrating complex ({protomer.smiles}, charge {protomer.charge:+d})")
    topology, positions = equilibrate(
        topology, positions, forcefield, config=config.equilibration_config
    )
    equilibrated_pdb = work_dir / "equilibrated.pdb"
    PDBFile.writeFile(topology, positions, equilibrated_pdb.open("w"), keepIds=True)
    return _PreparedComplex(
        protomer=protomer,
        topology=topology,
        positions=positions,
        forcefield=forcefield,
        equilibrated_pdb=equilibrated_pdb,
        protomer_sdf=protomer_sdf,
        work_dir=work_dir,
    )


def _produce_and_score(
    prepared: _PreparedComplex,
    config: MMGBSASettings,
    run_time: unit.Quantity,
    tag: str,
) -> tuple[list[float], Path, np.ndarray]:
    """Run production MD of length ``run_time`` from the equilibrated state and score MMGBSA.

    ``production`` seeds a fresh context from the equilibrated topology/positions,
    so it can be called more than once on the same ``prepared`` complex (a short
    screen, then the full run) without state leaking between calls. Returns the
    per-frame interaction energies, the trajectory path, and the per-frame ligand
    RMSD (nm) to the first frame in the protein's reference frame.
    """
    run_ns = run_time.value_in_unit(unit.nanosecond)
    smiles = prepared.protomer.smiles
    traj_path = prepared.work_dir / f"production_{tag}.dcd"

    logger.info(f"Running {run_ns} ns {tag} MD ({smiles})")
    production(
        prepared.topology,
        prepared.positions,
        prepared.forcefield,
        traj_path,
        config=ProductionSettings(run_time=run_time, rest_ligand=True),
    )

    logger.info(f"Computing n-closest-waters MMGBSA ({smiles}, {tag})")
    energies, rmsd, _top, _traj = get_interaction_energy(
        pdb_path=str(prepared.equilibrated_pdb),
        ligand_path=str(prepared.protomer_sdf),
        traj_path=str(traj_path),
        close_traj_path=str(prepared.work_dir / f"mmgbsa_close_{tag}.dcd"),
        close_top_path=str(prepared.work_dir / f"mmgbsa_close_{tag}.pdb"),
        n_closest_waters=config.n_closest_waters,
        ligand_resname=config.ligand_resname,
    )
    return energies, traj_path, rmsd


def _closest_waters_complex(
    topology: Topology,
    positions: unit.Quantity,
    ligand_resname: str,
    n_closest_waters: int,
) -> Modeller:
    """Trim a solvated frame to protein + ligand + the n waters closest to the ligand.

    Mirrors the n-closest-waters system that MMGBSA scores, applied to the single
    representative frame: ions are dropped and only the ``n_closest_waters`` water
    molecules nearest any ligand atom are kept, with their real frame coordinates.
    """
    coords = np.asarray(positions.value_in_unit(unit.nanometer))
    positions = unit.Quantity([Vec3(*row) for row in coords], unit.nanometer)
    modeller = Modeller(topology, positions)

    lig_idxs = [a.index for a in modeller.topology.atoms() if a.residue.name == ligand_resname]
    lig_coords = coords[lig_idxs]

    water_residues: list = []
    water_o_idxs: list[int] = []
    ion_residues: list = []
    for res in modeller.topology.residues():
        if res.name in _WATER_RESNAMES:
            atoms = list(res.atoms())
            o_atom = next(
                (a for a in atoms if a.element is not None and a.element.symbol == "O"), atoms[0]
            )
            water_residues.append(res)
            water_o_idxs.append(o_atom.index)
        elif res.name in _ION_RESNAMES:
            ion_residues.append(res)

    to_delete = list(ion_residues)
    if water_residues and len(lig_idxs) > 0:
        dmin = cdist(coords[water_o_idxs], lig_coords).min(axis=1)
        keep = {int(i) for i in np.argsort(dmin)[:n_closest_waters]}
        to_delete += [res for i, res in enumerate(water_residues) if i not in keep]

    modeller.delete(to_delete)
    return modeller


def _ligand_residue_contacts(
    traj: md.Trajectory, ligand_resname: str, cutoff: float = 0.4
) -> tuple[dict[str, int], int]:
    """Per-protein-residue ligand-contact frame counts over ``traj``.

    A protein residue counts as in contact in a frame when any of its heavy atoms
    lies within ``cutoff`` nm (default 4 A) of any ligand heavy atom; distances use
    the minimum image so a PBC-wrapped ligand still registers. Returns
    ``{residue_label: n_frames_in_contact}`` keyed by the residue's PDB label
    (name + original number, e.g. ``ASP189``; chain-prefixed when the protein spans
    more than one chain) alongside ``traj.n_frames``, so callers can pool counts
    across replicas before converting to a fraction. Residues never in contact
    are omitted.
    """
    lig_heavy = np.array(
        [
            a.index
            for a in traj.topology.atoms
            if a.residue.name == ligand_resname
            and a.element is not None
            and a.element.symbol != "H"
        ]
    )
    if lig_heavy.size == 0:
        return {}, traj.n_frames

    prot_residues = [r for r in traj.topology.residues if r.is_protein]
    multi_chain = len({r.chain.index for r in prot_residues}) > 1
    counts: dict[str, int] = {}
    for res in prot_residues:
        res_heavy = np.array(
            [a.index for a in res.atoms if a.element is not None and a.element.symbol != "H"]
        )
        if res_heavy.size == 0:
            continue
        neighbors = md.compute_neighbors(traj, cutoff, lig_heavy, haystack_indices=res_heavy)
        n_contact = int(sum(1 for frame_neighbors in neighbors if len(frame_neighbors) > 0))
        if n_contact == 0:
            continue
        label = f"{res.name}{res.resSeq}"
        if multi_chain:
            label = f"{res.chain.index}:{label}"
        counts[label] = n_contact
    return counts, traj.n_frames


def _lowest_energy_frame(
    prepared: _PreparedComplex, traj_path: Path, energies: list[float]
) -> tuple[Topology, unit.Quantity, int, int]:
    """Return the lowest-MMGBSA frame of ``traj_path`` as a full solvated system.

    Reuses the equilibrated OpenMM ``topology`` the forcefield already matched
    (so a ``System`` can be rebuilt from it downstream), with its box set to this
    frame's cell and the ligand imaged back beside the protein. Returns the
    ``(topology, positions, frame_index, n_frames)`` of the chosen frame.
    """
    traj = md.load(str(traj_path), top=str(prepared.equilibrated_pdb))
    best = int(np.argmin(energies))
    frame = traj[best]

    topology = prepared.topology
    if frame.unitcell_vectors is not None:
        topology.setPeriodicBoxVectors(frame.unitcell_vectors[0] * unit.nanometer)
    positions = _recenter_ligand_positions(topology, frame.xyz[0] * unit.nanometer)
    return topology, positions, best, traj.n_frames


def _snapshot_from_frame(
    topology: Topology,
    positions: unit.Quantity,
    ligand_rdmol: Chem.Mol,
    ligand_resname: str,
) -> SystemState:
    """Wrap the lowest-MMGBSA full solvated frame as a :class:`SystemState`.

    ``system`` is left ``None`` (the consumer rebuilds it with the right protomer,
    as ModBinddG does when it attaches escape restraints); ``ligand`` carries
    ``ligand_rdmol`` placed at this frame's ligand coordinates. The explicit
    topology keeps the ligand first and preserves the RDKit atom order, so a
    positional copy of its coordinates is correct.
    """
    lig_idxs = [a.index for a in topology.atoms() if a.residue.name == ligand_resname]
    coords_ang = np.array(positions.value_in_unit(unit.angstrom))
    ligand_out = set_coordinates(ligand_rdmol, coords=coords_ang[lig_idxs])
    return SystemState(topology=topology, positions=positions, ligand=ligand_out)


def _write_representative(
    topology: Topology,
    positions: unit.Quantity,
    ligand_rdmol: Chem.Mol,
    config: MMGBSASettings,
    prot_path: Path,
    lig_path: Path,
) -> None:
    """Split the lowest-MMGBSA frame into a protein+close-waters PDB and a ligand SDF."""
    modeller = _closest_waters_complex(
        topology, positions, config.ligand_resname, config.n_closest_waters
    )

    prot_modeller = Modeller(modeller.topology, modeller.positions)
    prot_modeller.delete(
        [a for a in prot_modeller.topology.atoms() if a.residue.name == config.ligand_resname]
    )

    lig_modeller = Modeller(modeller.topology, modeller.positions)
    lig_modeller.delete(
        [a for a in lig_modeller.topology.atoms() if a.residue.name != config.ligand_resname]
    )

    # The solvated topology places the ligand first and preserves the RDKit atom
    # order of ``ligand_rdmol`` (OpenFF -> OpenMM keep atom order), so the frame's
    # ligand coordinates map onto the RDKit mol positionally.
    lig_coords_ang = np.array(lig_modeller.positions.value_in_unit(unit.angstrom))
    ligand_out = set_coordinates(ligand_rdmol, coords=lig_coords_ang)

    PDBFile.writeFile(
        prot_modeller.topology,
        prot_modeller.positions,
        prot_path.open("w"),
        keepIds=True,
    )
    Chem.MolToMolFile(ligand_out, str(lig_path))


@dataclass
class MMGBSAResult:
    """Outputs of one MMGBSA funnel run.

    ``protein_path``/``ligand_path`` are the published representative
    protein-plus-close-waters PDB and ligand SDF (same scheme as the run's
    ``output``); ``scores`` is the MMGBSA score series; ``snapshot`` is the global
    lowest-MMGBSA frame as a full solvated :class:`SystemState` (topology with box,
    positions and the winning-protomer ligand placed at the frame pose) for
    downstream reuse - e.g. as ModBinddG's equilibrated escape starting structure.
    ``contacts`` maps each protein residue label to the fraction (0-1) of production
    frames (pooled over all replicas) in which it contacts the ligand.
    """

    protein_path: str
    ligand_path: str
    scores: pd.Series
    snapshot: SystemState
    contacts: dict[str, float]


def run_mmgbsa(
    protein: str,
    ligand: str,
    output: str,
    config: MMGBSASettings | None = None,
) -> MMGBSAResult:
    """Run an MMGBSA calculation for one protein-ligand pair.

    PDBFixer protonates the protein and uniKa enumerates the ligand's protomers
    within ``config.protonation_penalty`` of the solution-dominant one at the
    target pH. The protomers are funnelled from cheap to expensive:

    1. every protomer is minimised and scored in implicit solvent;
    2. the single best of those earns a full explicit run - solvation,
       equilibration and ``config.production_time`` production - scored by the
       n-closest-waters MMGBSA.

    At every stage a protomer is ranked by ``intrinsic_free_energy + MMGBSA`` -
    the bound-state free energy - so the pocket can shift the protonation state
    away from the solution-dominant one when binding pays for the intrinsic cost
    (e.g. an amidinium against an aspartate), without a blanket bias toward the
    most charged protomer. The published score and representative outputs
    (lowest-energy frame, split into protein-plus-close-waters and ligand) come
    from the winner's full explicit run.

    ``protein``, ``ligand`` and ``output`` may each be a local path or an
    ``s3://`` URI. Returns an :class:`MMGBSAResult` with the published protein and
    ligand locations (same scheme as ``output``), the MMGBSA score series, and the
    global lowest-energy frame as a reusable :class:`SystemState` snapshot.
    """
    if config is None:
        config = MMGBSASettings()

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

        # 1. Protonate the protein (shared by every protomer's complex).
        run_pdbfixer(local_protein, fixed_protein)
        protein_pdb = PDBFile(str(fixed_protein))
        protein_modeller = Modeller(protein_pdb.topology, protein_pdb.positions)

        # 2. Enumerate ligand protomers within the penalty window at the target pH.
        penalty_kcal = config.protonation_penalty.value_in_unit(unit.kilocalories_per_mole)
        protomers = _enumerate_protomers(local_ligand, config.protomer_ph, penalty_kcal, UnipKa())

        # Per-protomer bookkeeping for protomers.csv, keyed by enumeration order.
        records: list[dict] = [
            {
                "smiles": p.smiles,
                "charge": p.charge,
                "intrinsic_kcal": p.intrinsic_kcal,
                "implicit_min_kcal": float("nan"),
                "implicit_min_corrected": float("nan"),
                "selected": False,
            }
            for p in protomers
        ]

        # 3. Implicit filter: minimise + score every protomer in implicit solvent,
        #    then pick the single best corrected score for the full explicit run.
        implicit_ranked: list[tuple[float, int]] = []  # (corrected, protomer index)
        for i, protomer in enumerate(protomers):
            topology, positions, forcefield = _build_implicit_complex(protomer, protein_modeller)
            scorer = _ImplicitScorer(topology, positions, forcefield, config.ligand_resname)
            interaction = scorer.interaction_energy(scorer.minimize(positions))
            corrected = protomer.intrinsic_kcal + interaction
            records[i]["implicit_min_kcal"] = interaction
            records[i]["implicit_min_corrected"] = corrected
            implicit_ranked.append((corrected, i))
            logger.info(
                f"[implicit-min] {protomer.smiles} (charge {protomer.charge:+d}): "
                f"interaction {interaction:.2f} + intrinsic {protomer.intrinsic_kcal:.2f} "
                f"= corrected {corrected:.2f} kcal/mol"
            )

        winner_idx = min(implicit_ranked)[1]
        records[winner_idx]["selected"] = True
        winner = protomers[winner_idx]
        logger.info(
            f"Funnel selected {winner.smiles} (charge {winner.charge:+d}) for the full explicit run"
        )

        protomers_df = pd.DataFrame(records)
        protomers_df.to_csv(protomers_path, index=False)
        if len(records) > 1:
            logger.info(f"Protomer funnel:\n{protomers_df.to_string(index=False)}")

        # 5. Full explicit run for the winner only: solvate, equilibrate, produce.
        prepared = _prepare_and_equilibrate(winner, protein_modeller, config, tmp_dir / "winner")

        # Run ``n_replicas`` independent production replicas from the shared
        # equilibrated state (``production`` reseeds velocities each call). Their
        # per-frame energies are pooled for the score; the global lowest-energy
        # frame becomes the representative pose.
        replica_energies: list[list[float]] = []
        replica_trajs: list[Path] = []
        replica_rmsds: list[np.ndarray] = []
        contact_counts: dict[str, int] = {}
        contact_frames = 0
        for r in range(config.n_replicas):
            energies_r, traj_r, rmsd_r = _produce_and_score(
                prepared, config, config.production_time, f"production_r{r}"
            )
            replica_energies.append(energies_r)
            replica_trajs.append(traj_r)
            replica_rmsds.append(np.asarray(rmsd_r, dtype=float))

            # Pool this replica's protein-ligand contacts into the running totals.
            traj_r_md = md.load(str(traj_r), top=str(prepared.equilibrated_pdb))
            counts_r, n_frames_r = _ligand_residue_contacts(traj_r_md, config.ligand_resname)
            for label, n in counts_r.items():
                contact_counts[label] = contact_counts.get(label, 0) + n
            contact_frames += n_frames_r

            logger.info(
                f"Replica {r + 1}/{config.n_replicas} ({winner.smiles}) explicit MMGBSA "
                f"{np.mean(energies_r):.2f} kcal/mol over {len(energies_r)} frames"
            )

        winner_arr = np.asarray([e for es in replica_energies for e in es], dtype=float)
        # Mean ligand RMSD (A) to the first frame in the protein's reference frame
        # (pose drift in the pocket), pooled over all replicas.
        ligand_rmsd_mean = float(np.concatenate(replica_rmsds).mean() * 10.0)
        # Fraction of frames (over all replicas, 0-1) each protein residue contacts
        # the ligand, keyed by residue label and ordered by decreasing occupancy.
        contacts = {
            label: n / contact_frames
            for label, n in sorted(contact_counts.items(), key=lambda kv: kv[1], reverse=True)
        }
        corrected = winner.intrinsic_kcal + float(winner_arr.mean())
        logger.info(
            f"Winner {winner.smiles} (charge {winner.charge:+d}) explicit MMGBSA "
            f"{winner_arr.mean():.2f} + intrinsic {winner.intrinsic_kcal:.2f} "
            f"= corrected {corrected:.2f} kcal/mol "
            f"({config.n_replicas} replica(s), {winner_arr.size} frames)"
        )

        # 6. Score series and representative outputs come from the winning protomer.
        scores = pd.Series(
            {
                "interaction_energy": float(winner_arr.mean()),
                "interaction_energy_std": float(winner_arr.std()),
                "interaction_energy_min": float(winner_arr.min()),
                "n_frames": int(winner_arr.size),
                "n_replicas": int(config.n_replicas),
                "ligand_rmsd": ligand_rmsd_mean,
                "ligand_charge": int(winner.charge),
                "protonation_penalty": float(winner.intrinsic_kcal),
                "mmgbsa_score": float(corrected),
            }
        )
        scores.to_csv(score_path, header=False)
        logger.info(
            f"MMGBSA = {scores['interaction_energy']:.2f} "
            f"+/- {scores['interaction_energy_std']:.2f} kcal/mol over "
            f"{int(scores['n_frames'])} frames ({int(scores['n_replicas'])} replica(s))."
        )
        logger.info(
            f"Mean ligand RMSD to first frame {ligand_rmsd_mean:.2f} A over {contact_frames} frames"
        )
        if contacts:
            top = ", ".join(f"{lbl} {frac:.2f}" for lbl, frac in list(contacts.items())[:10])
            logger.info(f"Protein-ligand contacts (top {min(10, len(contacts))}): {top}")

        # Persist the per-residue contact occupancies alongside the scalar scores.
        contacts_path = tmp_dir / "contacts.csv"
        pd.DataFrame(
            {"residue": list(contacts), "contact_fraction": list(contacts.values())}
        ).to_csv(contacts_path, index=False)

        # The representative pose and the reusable escape snapshot are the global
        # lowest-energy frame, taken from whichever replica produced it.
        best_replica = int(np.argmin([min(es) for es in replica_energies]))
        topology, positions, best_frame, n_frames = _lowest_energy_frame(
            prepared, replica_trajs[best_replica], replica_energies[best_replica]
        )
        logger.info(
            f"Representative frame {best_frame}/{n_frames} from replica "
            f"{best_replica + 1}/{config.n_replicas} "
            f"(MMGBSA {min(replica_energies[best_replica]):.2f} kcal/mol)"
        )
        snapshot = _snapshot_from_frame(topology, positions, winner.mol, config.ligand_resname)
        _write_representative(topology, positions, winner.mol, config, prot_path, lig_path)

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
        (out_dir / "contacts.csv").write_bytes(contacts_path.read_bytes())

        logger.info(f"Saved scores to {out_scores}")
        logger.info(f"Saved representative protein to {out_prot}")
        logger.info(f"Saved representative ligand to {out_lig}")

    return MMGBSAResult(
        protein_path=str(out_prot),
        ligand_path=str(out_lig),
        scores=scores,
        snapshot=snapshot,
        contacts=contacts,
    )


@click.command()
@click.option("--protein", required=True, help="Protein PDB file (local path or s3:// URI).")
@click.option("--ligand", required=True, help="Ligand MOL/SDF file (local path or s3:// URI).")
@click.option("--output", required=True, help="Output directory (local path or s3:// prefix).")
@click.option(
    "--production-time",
    default=0.5,
    show_default=True,
    help="Full explicit production MD time (ns) for the winning protomer.",
)
@click.option(
    "--n-replicas",
    default=3,
    show_default=True,
    help="Independent production replicas for the winning protomer (pooled for the score).",
)
@click.option(
    "--ph",
    default=7.0,
    show_default=True,
    help="pH at which the ligand protomers are enumerated.",
)
@click.option(
    "--protonation-penalty",
    default=3.0,
    show_default=True,
    help="Enumerate every protomer within this uniKa free-energy window (kcal/mol).",
)
@click.option("--n-closest-waters", default=5, show_default=True, help="Explicit waters to keep.")
def main(
    protein: str,
    ligand: str,
    output: str,
    production_time: float,
    n_replicas: int,
    ph: float,
    protonation_penalty: float,
    n_closest_waters: int,
) -> None:
    """Run an MMGBSA calculation from the command line."""
    config = MMGBSASettings(
        production_time=production_time * unit.nanosecond,
        n_replicas=n_replicas,
        protomer_ph=ph,
        protonation_penalty=protonation_penalty * unit.kilocalories_per_mole,
        n_closest_waters=n_closest_waters,
    )
    result = run_mmgbsa(protein, ligand, output, config=config)
    logger.info(f"Scores:\n{result.scores}")
    logger.info(f"Protein: {result.protein_path}")
    logger.info(f"Ligand: {result.ligand_path}")


if __name__ == "__main__":
    main()
