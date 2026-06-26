#!/usr/bin/env python3
"""Run cpptraj Grid Inhomogeneous Solvation Theory (GIST) on a trajectory.

GIST needs an Amber topology with Lennard-Jones parameters. OpenMM/OpenFF
trajectories ship as PDB + DCD, so this script rebuilds the force field from
the ligand SDF, exports a prmtop via ParmEd, writes a cpptraj input file, and
runs cpptraj.

Example:
    pixi run python scripts/run_cpptraj_gist.py \\
        ~/Desktop/cph_holo/trajectories/replica_0000.0_0.pdb \\
        ~/Desktop/cph_holo/trajectories/replica_0000.0_0.dcd
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import mdtraj as md
import numpy as np
from openff.toolkit.topology import Molecule
from openmm import app, unit
from parmed import openmm as pmd_omm
from parmed.amber import AmberParm
from parmed.topologyobjects import AtomType, BondType

from opensqm.md.prepare import _PROTEIN_FORCEFIELD_FILES, get_ligand_forcefield

DEFAULT_PDB = Path("~/Desktop/cph_holo/trajectories/replica_0000.0_0.pdb").expanduser()
DEFAULT_DCD = Path("~/Desktop/cph_holo/trajectories/replica_0000.0_0.dcd").expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdb",
        nargs="?",
        type=Path,
        default=DEFAULT_PDB,
        help=f"Topology PDB (default: {DEFAULT_PDB}).",
    )
    parser.add_argument(
        "dcd",
        nargs="?",
        type=Path,
        default=DEFAULT_DCD,
        help=f"Trajectory DCD (default: {DEFAULT_DCD}).",
    )
    parser.add_argument(
        "--ligand-sdf",
        type=Path,
        default=None,
        help="Ligand SDF for OpenFF parameterisation. "
        "Defaults to <run_dir>/ligand_variants/LIG.sdf when present.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for prmtop, cpptraj input, and GIST outputs "
        "(default: same directory as the PDB).",
    )
    parser.add_argument(
        "--prefix",
        default="gist",
        help="Output prefix for GIST grids and tables (default: gist).",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=0.5,
        help="Voxel spacing in Angstrom (default: 0.5).",
    )
    parser.add_argument(
        "--grid-padding",
        type=float,
        default=8.0,
        help="Padding around the ligand bounding box in Angstrom (default: 8).",
    )
    parser.add_argument(
        "--grid-center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Grid centre in Angstrom. Default: ligand centre of mass.",
    )
    parser.add_argument(
        "--griddim",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Grid dimensions in voxels. Default: fit ligand + padding.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="First trajectory frame (cpptraj 1-based, default: 1).",
    )
    parser.add_argument(
        "--stop-frame",
        type=int,
        default=0,
        help="Last trajectory frame; 0 means all frames (default: 0).",
    )
    parser.add_argument(
        "--ligand-mask",
        default=":LIG",
        help="cpptraj mask for the ligand (default: :LIG).",
    )
    parser.add_argument(
        "--prmtop",
        type=Path,
        default=None,
        help="Existing Amber prmtop. Skips force-field export when provided.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Write cpptraj input only; do not execute cpptraj.",
    )
    parser.add_argument(
        "--cpptraj",
        default="cpptraj",
        help="cpptraj executable (default: cpptraj).",
    )
    return parser.parse_args()


def _resolve_ligand_sdf(pdb: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    run_dir = pdb.parent.parent
    candidate = run_dir / "ligand_variants" / "LIG.sdf"
    if candidate.exists():
        return candidate.resolve()
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        import json

        ligand = json.loads(manifest.read_text()).get("ligand")
        if ligand:
            path = Path(ligand).expanduser()
            if path.exists():
                return path.resolve()
    raise FileNotFoundError(
        "Could not locate a ligand SDF. Pass --ligand-sdf explicitly."
    )


def _sigma_to_rmin(sigma: float) -> float:
    return sigma * (2.0 ** (-1.0 / 6.0)) if sigma else 0.0


def export_prmtop(pdb: Path, ligand_sdf: Path, prmtop_out: Path) -> Path:
    """Build an Amber prmtop from the PDB using the OpenMM/OpenFF force field."""
    if prmtop_out.exists():
        return prmtop_out

    offmol = Molecule.from_file(str(ligand_sdf))
    forcefield = get_ligand_forcefield([offmol])
    forcefield.loadFile(_PROTEIN_FORCEFIELD_FILES)

    pdb_file = app.PDBFile(str(pdb))
    system = forcefield.createSystem(
        pdb_file.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=9.0 * unit.angstroms,
        constraints=app.HBonds,
        rigidWater=True,
    )
    struct = pmd_omm.load_topology(
        pdb_file.topology,
        xyz=pdb_file.positions,
        system=system,
    )

    type_map: dict[str, AtomType] = {}
    next_num = 1
    for atom in struct.atoms:
        key = str(atom.type)[:4]
        if key not in type_map:
            atom_type = AtomType(key, next_num, atom.mass, atom.atomic_number)
            atom_type.set_lj_params(atom.epsilon, _sigma_to_rmin(atom.sigma))
            type_map[key] = atom_type
            next_num += 1
        atom.type = key

    dummy_bond = BondType(1.0, 100.0)
    struct.bond_types.append(dummy_bond)
    for bond in struct.bonds:
        bond.type = dummy_bond

    struct.dihedrals.clear()
    struct.impropers.clear()
    struct.angles.clear()
    struct.urey_bradleys.clear()
    struct.cmaps.clear()
    struct.adjusts.clear()

    prmtop_out.parent.mkdir(parents=True, exist_ok=True)
    AmberParm.from_structure(struct).write_parm(str(prmtop_out))
    return prmtop_out


def _grid_from_ligand(
    pdb: Path,
    spacing: float,
    padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    traj = md.load_pdb(str(pdb))
    ligand_atoms = traj.topology.select("resname LIG")
    if ligand_atoms.size == 0:
        raise RuntimeError("No :LIG residues found in the topology PDB.")
    coords_a = traj.xyz[0, ligand_atoms] * 10.0
    center = coords_a.mean(axis=0)
    span = coords_a.max(axis=0) - coords_a.min(axis=0)
    griddim = np.ceil((span + 2.0 * padding) / spacing).astype(int)
    griddim = np.maximum(griddim, 20)
    return center, griddim


def write_cpptraj_input(
    *,
    prmtop: Path,
    dcd: Path,
    output_dir: Path,
    prefix: str,
    grid_center: np.ndarray,
    griddim: np.ndarray,
    grid_spacing: float,
    ligand_mask: str,
    start_frame: int,
    stop_frame: int,
) -> Path:
    if stop_frame > 0:
        frame_range = f"{start_frame} {stop_frame}"
    elif start_frame > 1:
        frame_range = str(start_frame)
    else:
        frame_range = ""

    cx, cy, cz = grid_center
    nx, ny, nz = griddim
    gist_out = output_dir / f"{prefix}.dat"

    lines = [
        f"parm {prmtop}",
        f"trajin {dcd}" + (f" {frame_range}" if frame_range else ""),
        f"autoimage anchor {ligand_mask} familiar",
        (
            f"gist gridcntr {cx:.3f} {cy:.3f} {cz:.3f} "
            f"griddim {nx} {ny} {nz} gridspacn {grid_spacing:.3f} "
            f"prefix {prefix} out {gist_out.name}"
        ),
        "go",
    ]

    input_path = output_dir / f"{prefix}.in"
    input_path.write_text("\n".join(lines) + "\n")
    return input_path


def run_cpptraj(cpptraj: str, input_path: Path, output_dir: Path) -> None:
    log_path = output_dir / f"{input_path.stem}.log"
    with log_path.open("w") as log_handle:
        subprocess.run(
            [cpptraj, "-i", str(input_path)],
            cwd=output_dir,
            check=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    print(f"Wrote log to {log_path}")


def main() -> None:
    args = parse_args()
    pdb = args.pdb.expanduser().resolve()
    dcd = args.dcd.expanduser().resolve()
    output_dir = (args.output_dir or pdb.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdb.exists():
        raise FileNotFoundError(pdb)
    if not dcd.exists():
        raise FileNotFoundError(dcd)

    if args.prmtop is not None:
        prmtop = args.prmtop.expanduser().resolve()
    else:
        ligand_sdf = _resolve_ligand_sdf(pdb, args.ligand_sdf)
        prmtop = output_dir / f"{pdb.stem}.prmtop"
        print(f"Exporting prmtop from {ligand_sdf.name} ...")
        export_prmtop(pdb, ligand_sdf, prmtop)
        print(f"Wrote {prmtop}")

    auto_center, auto_griddim = _grid_from_ligand(pdb, args.grid_spacing, args.grid_padding)
    center = (
        np.asarray(args.grid_center, dtype=float)
        if args.grid_center is not None
        else auto_center
    )
    griddim = (
        np.asarray(args.griddim, dtype=int)
        if args.griddim is not None
        else auto_griddim
    )

    input_path = write_cpptraj_input(
        prmtop=prmtop,
        dcd=dcd,
        output_dir=output_dir,
        prefix=args.prefix,
        grid_center=center,
        griddim=griddim,
        grid_spacing=args.grid_spacing,
        ligand_mask=args.ligand_mask,
        start_frame=args.start_frame,
        stop_frame=args.stop_frame,
    )
    print(f"Wrote {input_path}")
    print(
        f"Grid centre (Å): {center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}; "
        f"dimensions: {griddim[0]} x {griddim[1]} x {griddim[2]} "
        f"@ {args.grid_spacing} Å"
    )

    if args.skip_run:
        print("Skipping cpptraj execution (--skip-run).")
        return

    print("Running cpptraj GIST ...")
    run_cpptraj(args.cpptraj, input_path, output_dir)
    print(f"GIST outputs written under {output_dir} (prefix: {args.prefix})")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"cpptraj failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
