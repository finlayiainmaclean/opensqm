"""MMGBSA interaction energy from a constant-pH run at the target pH.

A constant-pH MD run over a tight pH-REMD ladder of +/-2 pH units around pH 7
(one fixed-pH replica per rung) samples the protein-ligand complex in the
protonation ensemble appropriate for each pH. Because a ligand is present,
``run_cph`` computes the n-closest-waters MMGBSA interaction energy over the
trajectory (``compute_replica_mmgbsa``, written per pH to ``mmgbsa/by_ph.csv``) -
so that IS the score, and no separate production MD is run. Scoring and the
snapshot are taken from the pH-7 stratum: the lowest-energy frame of its dominant
macrostate is the representative snapshot, split into a protein (plus its closest
waters) PDB and a protonated-ligand SDF in the frame pose.

Inputs (protein, ligand) and the output location may each be a local path or an
``s3://`` URI: inputs are staged into a temp dir, all work happens locally, and
only the three published artifacts are copied/uploaded to the output location.
"""

import tempfile
from pathlib import Path

import click
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath
from loguru import logger
from openmm import Vec3, unit
from openmm.app import Modeller
from openmm.app.pdbfile import PDBFile
from pydantic import BaseModel, ConfigDict
from pydantic_units import OpenMMQuantity
from rdkit import Chem, RDLogger
from scipy.spatial.distance import cdist

from opensqm.cph.run_cph import ConstantpHRunSettings, PHResult, SystemState, run_cph
from opensqm.fix import run_pdbfixer
from opensqm.rdkit_utils import set_coordinates

RDLogger.DisableLog("rdApp.warning")

_WATER_RESNAMES = ("HOH", "WAT", "TIP3", "TIP4", "TIP5")
_ION_RESNAMES = ("NA", "CL")

# MMGBSA is defined at pH 7. A tight pH-REMD ladder of +/-2 pH units around it
# improves protonation-state sampling (replicas swap pH, shuttling macrostates
# onto the pH-7 replica) while keeping strong swap overlap; scoring and the
# snapshot are taken from the pH-7 stratum.
_MMGBSA_PH = 7.0
# +/-4 pH units around the target, one fixed-pH replica per rung.
_MMGBSA_PH_LADDER = [_MMGBSA_PH - 4, _MMGBSA_PH - 2, _MMGBSA_PH + 0, _MMGBSA_PH + 2, _MMGBSA_PH + 4]


class MMGBSASettings(BaseModel):
    """Settings for the CpH MMGBSA calculation."""

    model_config = ConfigDict(frozen=True)
    n_closest_waters: int = 5
    ligand_resname: str = "LIG"
    # Constant-pH MD time per replica. This is the only sampling: MMGBSA is scored
    # over this trajectory and its lowest-energy frame becomes the representative pose.
    cph_production_time: OpenMMQuantity[unit.nanosecond] = 0.5 * unit.nanosecond


def _cph_snapshot(
    protein: Path,
    ligand: Path,
    cph_output: Path,
    config: MMGBSASettings,
) -> SystemState:
    """Run pH-REMD CpH around pH 7 and return its representative (lowest-energy) snapshot.

    ``run_cph`` sees a ligand and therefore computes the n-closest-waters MMGBSA
    over the trajectory (writing ``mmgbsa/by_ph.csv``). Sampling uses a tight
    pH-REMD ladder around pH 7; the returned snapshot is the representative
    frame of the pH-7 replica, whose ligand is captured in the dominant
    protonation state at pH 7.
    """
    production_ns = config.cph_production_time.value_in_unit(unit.nanosecond)
    logger.info(f"Running {production_ns} ns CpH with pH-REMD ladder {_MMGBSA_PH_LADDER}")
    cph_result = run_cph(
        protein,
        output=str(cph_output),
        ligand=ligand,
        config=ConstantpHRunSettings(
            ph=_MMGBSA_PH_LADDER,
            target_ph=_MMGBSA_PH,
            production_time=config.cph_production_time,
            use_ph_remd=True,
            # One fixed-pH replica per rung (pH 5/7/9); no simulated-tempering
            # weights to optimise. The ligand protomers are enumerated at the
            # target pH (target_ph) rather than across the ladder.
            simulated_annealing=False,
            mmgbsa_n_closest_waters=config.n_closest_waters,
            protonation_penalty=3.0 * unit.kilocalories_per_mole,
            titratable_residue_query="(protein within 5 of resn LIG) or (resn LIG)",
        ),
        resume=False,
    )
    ph_results: list[PHResult] = cph_result["ph_results"]
    ph_result = next(
        (r for r in ph_results if abs(r.ph - _MMGBSA_PH) < 1e-6),
        None,
    )
    if ph_result is None:
        raise RuntimeError(f"CpH produced no pH-{_MMGBSA_PH} result to score")
    logger.info(f"Dominant macrostate population at pH {_MMGBSA_PH}: {ph_result.population:.3f}")

    snapshot = ph_result.lowest_energy_snapshot
    if snapshot is None or snapshot.ligand is None:
        raise RuntimeError("CpH did not yield a ligand-bearing snapshot")
    return snapshot


def _closest_waters_complex(
    snapshot: SystemState, ligand_resname: str, n_closest_waters: int
) -> Modeller:
    """Trim the solvated snapshot to protein + ligand + the n waters closest to it.

    Mirrors the n-closest-waters system that MMGBSA scores, applied to the single
    representative frame: ions are dropped and only the ``n_closest_waters`` water
    molecules nearest any ligand atom are kept, with their real frame coordinates.
    """
    coords = np.asarray(snapshot.positions.value_in_unit(unit.nanometer))
    positions = unit.Quantity([Vec3(*row) for row in coords], unit.nanometer)
    modeller = Modeller(snapshot.topology, positions)

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


def _ligand_frame_coords(ligand: Chem.Mol, name_to_coord: dict[str, np.ndarray]) -> np.ndarray:
    """Frame coordinates for ``ligand`` in ITS atom order, keyed by PDB name.

    The frame's trimmed LIG topology and the RDKit variant ``ligand`` contain
    the same atoms in *different* orders, so pulling coordinates positionally
    would scramble the pose. Both carry consistent PDB atom names, so map each
    RDKit atom to its frame coordinate by name. Raises if any atom cannot be
    matched (rather than silently emitting a wrong geometry).
    """
    coords = np.empty((ligand.GetNumAtoms(), 3), dtype=float)
    for i, atom in enumerate(ligand.GetAtoms()):
        info = atom.GetPDBResidueInfo()
        name = info.GetName().strip() if info is not None else None
        if name is None or name not in name_to_coord:
            raise ValueError(
                f"Ligand atom {i} (PDB name {name!r}) has no matching atom in the "
                "frame topology; cannot place the CpH ligand into the frame pose by name"
            )
        coords[i] = name_to_coord[name]
    return coords


def _read_mmgbsa_scores(cph_output: Path) -> pd.Series:
    """Read the pH-7 n-closest-waters MMGBSA stratum as the MMGBSA score.

    Under the pH-REMD ladder the score must come from the pH-7 stratum
    (``by_ph.csv``), not the ladder-wide ``overall.csv`` average.
    """
    by_ph_csv = cph_output / "mmgbsa" / "by_ph.csv"
    if not by_ph_csv.exists():
        raise FileNotFoundError(f"CpH MMGBSA summary not found: {by_ph_csv}")
    by_ph = pd.read_csv(by_ph_csv)
    row = by_ph.iloc[(by_ph["ph"] - _MMGBSA_PH).abs().argmin()]
    if abs(float(row["ph"]) - _MMGBSA_PH) > 1e-6:
        raise ValueError(
            f"No pH-{_MMGBSA_PH} MMGBSA stratum in {by_ph_csv} (found pHs "
            f"{sorted(by_ph['ph'].tolist())})"
        )
    return pd.Series(
        {
            "mm_energy_mean": float(row["mmgbsa_mean"]),
            "mm_energy_std": float(row["mmgbsa_std"]),
            "mm_energy_min": float(row["mmgbsa_min"]),
            "mm_energy_n_frames": int(row["mmgbsa_n_frames"]),
        }
    )


def _read_joint_microstate_mmgbsa(cph_output: Path) -> pd.DataFrame | None:
    """Read the per-joint-microstate MMGBSA table written by the CpH run, if present.

    Rows are joint protonation microstate labels (ligand protomer + each
    titratable residue variant, in the run's original PDB numbering); columns are
    the n-closest-waters MMGBSA interaction-energy summary (``mmgbsa_mean``/
    ``mmgbsa_std``/``mmgbsa_min`` in kcal/mol and the backing ``mmgbsa_n_frames``).
    MMGBSA is pH-invariant for a fixed microstate, so each microstate's frames are
    pooled across all pH strata into one estimate; the pH dependence of the score
    lives in the populations. Returns ``None`` when the CpH run produced no table.
    """
    csv_path = cph_output / "mmgbsa" / "by_joint_microstate.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path, index_col=0)


def run_mmgbsa(
    protein: str,
    ligand: str,
    output: str,
    config: MMGBSASettings | None = None,
) -> tuple[str, str, pd.Series]:
    """Run a CpH MMGBSA calculation for one protein-ligand pair.

    A constant-pH MD run over a +/-2 pH ladder around pH 7 (one fixed-pH replica
    per rung) samples the complex, picks the dominant ligand protonation state at
    the target pH, and (because a ligand is present) computes the n-closest-waters
    MMGBSA over the trajectory. The score comes straight from the pH-7 stratum of
    that CpH MMGBSA summary; the representative outputs are the lowest-energy frame
    of the dominant macrostate, split into protein-plus-close-waters and ligand.

    ``protein``, ``ligand`` and ``output`` may each be a local path or an
    ``s3://`` URI. Returns the published protein and ligand locations (same scheme
    as ``output``) and the MMGBSA score series.
    """
    if config is None:
        config = MMGBSASettings()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)

        tmp_dir = Path("/tmp/cph")

        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Stage inputs locally (downloading from S3 when needed). All heavy work
        # runs in the temp dir; only the three artifacts below are published.
        protein_src, ligand_src = AnyPath(protein), AnyPath(ligand)
        local_protein = tmp_dir / f"protein_input{protein_src.suffix or '.pdb'}"
        local_protein.write_bytes(protein_src.read_bytes())
        local_ligand = tmp_dir / f"ligand_input{ligand_src.suffix or '.sdf'}"
        local_ligand.write_bytes(ligand_src.read_bytes())

        fixed_protein = tmp_dir / "protein_prepared.pdb"
        cph_output = tmp_dir / "cph_equilibration"
        prot_path = tmp_dir / "prot.pdb"
        lig_path = tmp_dir / "lig.sdf"
        score_path = tmp_dir / "scores.csv"

        run_pdbfixer(local_protein, fixed_protein)

        # The +/-2 pH ladder CpH run is the whole calculation: it samples the complex
        # and scores MMGBSA over its trajectory. The snapshot is the representative
        # frame of the pH-7 stratum.
        snapshot = _cph_snapshot(fixed_protein, local_ligand, cph_output, config)

        # Representative complex frame trimmed to protein + ligand + n closest waters.
        modeller = _closest_waters_complex(snapshot, config.ligand_resname, config.n_closest_waters)

        prot_modeller = Modeller(positions=modeller.positions, topology=modeller.topology)
        prot_modeller.delete(
            [a for a in prot_modeller.topology.atoms() if a.residue.name == config.ligand_resname]
        )

        lig_modeller = Modeller(positions=modeller.positions, topology=modeller.topology)
        lig_modeller.delete(
            [a for a in lig_modeller.topology.atoms() if a.residue.name != config.ligand_resname]
        )

        # Ligand in the CpH-chosen protonation state, placed at the frame
        # conformation. The trimmed LIG topology and ``snapshot.ligand`` hold
        # the same atoms but NOT in the same order: constant-pH builds the
        # topology from the union super-template, where each titratable proton
        # is interleaved just after its parent, whereas the RDKit variant mol
        # keeps hydrogens in a trailing block. A positional copy would scramble
        # the geometry, so match atoms by PDB name (both carry consistent names
        # assigned in ``build_protonation_states``).
        lig_coords_ang = np.array(lig_modeller.positions.value_in_unit(unit.angstrom))
        name_to_coord = {
            atom.name: lig_coords_ang[i] for i, atom in enumerate(lig_modeller.topology.atoms())
        }
        ligand_coords = _ligand_frame_coords(snapshot.ligand, name_to_coord)
        ligand_rdmol = set_coordinates(snapshot.ligand, coords=ligand_coords)

        PDBFile.writeFile(
            prot_modeller.topology,
            prot_modeller.positions,
            prot_path.open("w"),
            keepIds=True,
        )
        Chem.MolToMolFile(ligand_rdmol, str(lig_path))

        scores = _read_mmgbsa_scores(cph_output)
        scores.to_csv(score_path, header=False)

        # Per-joint-microstate MMGBSA (ligand protomer x titratable residue
        # variants, per pH). Logged for inspection and published alongside the
        # scalar score when the CpH run produced it.
        joint_mmgbsa = _read_joint_microstate_mmgbsa(cph_output)
        if joint_mmgbsa is not None and not joint_mmgbsa.empty:
            logger.info(
                "MMGBSA per joint microstate (kcal/mol, pooled over pH):\n"
                f"{joint_mmgbsa.to_string()}"
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

        joint_csv = cph_output / "mmgbsa" / "by_joint_microstate.csv"
        if joint_csv.exists():
            (out_dir / "mmgbsa_by_microstate.csv").write_bytes(joint_csv.read_bytes())

        logger.info(f"Saved scores to {out_scores}")
        logger.info(f"Saved representative protein to {out_prot}")
        logger.info(f"Saved representative ligand to {out_lig}")

    return str(out_prot), str(out_lig), scores


@click.command()
@click.option("--protein", required=True, help="Protein PDB file (local path or s3:// URI).")
@click.option("--ligand", required=True, help="Ligand MOL/SDF file (local path or s3:// URI).")
@click.option("--output", required=True, help="Output directory (local path or s3:// prefix).")
@click.option(
    "--production-time",
    default=0.5,
    show_default=True,
    help="Constant-pH MD time (ns)",
)
@click.option("--n-closest-waters", default=5, show_default=True, help="Explicit waters to keep.")
def main(
    protein: str,
    ligand: str,
    output: str,
    production_time: float,
    n_closest_waters: int,
) -> None:
    """Run a CpH MMGBSA calculation from the command line."""
    config = MMGBSASettings(
        cph_production_time=production_time * unit.nanosecond,
        n_closest_waters=n_closest_waters,
    )
    protein, ligand, scores = run_mmgbsa(protein, ligand, output, config=config)
    logger.info(f"Scores:\n{scores}")
    logger.info(f"Protein: {protein}")
    logger.info(f"Ligand: {ligand}")


if __name__ == "__main__":
    main()
