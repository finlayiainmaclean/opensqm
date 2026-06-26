"""Parent class shared by grid-based and site-based water analysis.

This is a faithful, dependency-light reimplementation of the ``WaterAnalysis``
base class from SSTMap (Haider et al., *J. Chem. Theory Comput.* 2017,
DOI:10.1021/acs.jctc.7b00592). It is responsible for everything that grid-based
(GIST) and (potential future) site-based analyses have in common:

* reading the topology and partitioning atoms into water / solute groups,
* building the non-bonded parameter matrices (charges and Lennard-Jones
  coefficients) used for the pairwise energy evaluation,
* assigning hydrogen-bond donor/acceptor types and counting hydrogen bonds.

Only the parts required for grid water analysis are implemented; the public
behaviour matches the original where it overlaps.
"""

from __future__ import annotations

import os

import mdtraj as md
import numpy as np
import parmed as pmd

# Charge unit conversion: Amber stores charges such that q (in e) * 18.2223 makes
# q_i * q_j / r come out in kcal/mol (18.2223**2 ~= 332.06, Coulomb's constant).
CHARGE_CONVERSION = 18.2223

DON_ACC_ELEMENTS = ("oxygen", "nitrogen", "sulfur")
WATER_RESNAMES = (
    "H2O", "HHO", "OHH", "HOH", "OH2", "SOL", "WAT",
    "TIP", "TIP2", "TIP3", "TIP4", "T3P", "T4P", "T5P",
)
# 30 degrees in radians: the donor-H...acceptor angle cutoff for a hydrogen bond.
ANGLE_CUTOFF_RAD = 0.523599


class WaterAnalysis:
    """Base class that loads a trajectory's topology and force-field parameters.

    Parameters
    ----------
    topology_file : str
        Topology file (e.g. an Amber ``.prmtop`` / ``.parm7``).
    trajectory : str
        Molecular dynamics trajectory readable by mdtraj with ``top=topology_file``
        (e.g. an Amber NetCDF file).
    supporting_file : str, optional
        File providing non-bonded parameters when the topology itself does not
        (e.g. a PSF needs a CHARMM parameter directory). For Amber topologies the
        topology file is used directly.
    """

    def __init__(self, topology_file: str, trajectory: str, supporting_file: str | None = None):
        if not os.path.exists(topology_file) or not os.path.exists(trajectory):
            raise OSError(f"File {topology_file} or {trajectory} does not exist.")
        self.topology_file = topology_file
        self.trajectory = trajectory
        self.supporting_file = supporting_file if supporting_file is not None else topology_file
        self.comb_rule = "lorentz-bertholot"

        first_frame = md.load_frame(self.trajectory, 0, top=self.topology_file)
        if first_frame.unitcell_lengths is None:
            raise ValueError("Could not detect unit cell information in the trajectory.")
        self.topology = first_frame.topology

        self._build_atom_indices()

        self.chg_product, self.acoeff, self.bcoeff = self.generate_nonbonded_params()
        assert self.chg_product.shape == self.acoeff.shape == self.bcoeff.shape, (
            "Mismatch in non-bonded parameter matrices."
        )

        self.don_H_pair_dict: dict[int, list[list[int]]] = {}
        self.prot_hb_types = np.zeros(len(self.all_atom_ids), dtype=np.int64)
        self.solute_acc_ids, self.solute_don_ids, self.solute_acc_don_ids = self.assign_hb_types()

    def _build_atom_indices(self) -> None:
        """Partition atom indices into water, protein and other-solute groups."""
        top = self.topology
        self.all_atom_ids = top.select("all")
        self.prot_atom_ids = top.select("protein")
        self.wat_atom_ids = top.select("water")
        if self.wat_atom_ids.shape[0] == 0:
            expr = " or ".join(f"resname {name}" for name in WATER_RESNAMES)
            self.wat_atom_ids = top.select(expr)
        if self.wat_atom_ids.shape[0] == 0:
            raise ValueError("Unable to recognize water residues in the system!")
        if top.atom(self.wat_atom_ids[0]).name != "O":
            raise ValueError("Water oxygen must be the first atom of each water molecule.")

        self.wat_oxygen_atom_ids = np.asarray(
            [a for a in self.wat_atom_ids if top.atom(a).name == "O"]
        )
        # Number of sites per water molecule (3 for TIP3P, 4 for TIP4P, ...).
        self.water_sites = int(self.wat_oxygen_atom_ids[1] - self.wat_oxygen_atom_ids[0])
        for i in self.wat_oxygen_atom_ids:
            names = (top.atom(i).name[0], top.atom(i + 1).name[0], top.atom(i + 2).name[0])
            if names != ("O", "H", "H"):
                raise ValueError(
                    "Water molecules must be ordered as Oxygen, Hydrogen, Hydrogen, "
                    "Virtual-sites."
                )

        self.non_water_atom_ids = np.setdiff1d(self.all_atom_ids, self.wat_atom_ids)
        self.non_prot_atom_ids = np.setdiff1d(self.non_water_atom_ids, self.prot_atom_ids)
        if self.prot_atom_ids.shape[0] == 0:
            # No protein: treat every non-water atom as the "solute" for energies.
            self.prot_atom_ids = self.non_water_atom_ids
        assert (
            self.wat_atom_ids.shape[0] + self.non_water_atom_ids.shape[0]
            == self.all_atom_ids.shape[0]
        ), "Failed to partition atom indices in the system correctly!"

    def generate_nonbonded_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the per-(water-site, atom) non-bonded parameter matrices.

        Returns
        -------
        chg_product : numpy.ndarray, shape (n_sites, n_atoms)
            Product of the water-site charge with every atom charge (scaled so
            that ``chg_product / r`` is the electrostatic energy in kcal/mol).
        acoeff : numpy.ndarray, shape (n_sites, n_atoms)
            Lennard-Jones A coefficient ``4 eps sigma^12``.
        bcoeff : numpy.ndarray, shape (n_sites, n_atoms)
            Lennard-Jones B coefficient ``4 eps sigma^6``.
        """
        parm = pmd.load_file(self.supporting_file)
        n_atoms = self.all_atom_ids.shape[0]
        chg = np.empty(n_atoms)
        vdw = np.empty((n_atoms, 2))
        for i, at in enumerate(self.all_atom_ids):
            atom = parm.atoms[at]
            chg[i] = atom.charge
            vdw[i] = (atom.sigma, atom.epsilon)
        chg *= CHARGE_CONVERSION

        sites = slice(0, self.water_sites)
        water_chg = chg[self.wat_atom_ids[sites]].reshape(self.water_sites, 1)
        chg_product = water_chg * np.tile(chg, (self.water_sites, 1))

        water_sig = vdw[self.wat_atom_ids[sites], 0].reshape(self.water_sites, 1)
        water_eps = vdw[self.wat_atom_ids[sites], 1].reshape(self.water_sites, 1)
        if self.comb_rule == "geometric":
            mixed_sig = np.sqrt(water_sig * vdw[:, 0])
        else:  # lorentz-bertholot
            mixed_sig = 0.5 * (water_sig + vdw[:, 0])
        mixed_eps = np.sqrt(water_eps * vdw[:, 1])

        acoeff = 4 * mixed_eps * (mixed_sig**12)
        bcoeff = 4 * mixed_eps * (mixed_sig**6)
        return chg_product, acoeff, bcoeff

    def assign_hb_types(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classify solute atoms as hydrogen-bond acceptors, donors or both.

        Returns
        -------
        solute_acc_ids, solute_don_ids, solute_acc_don_ids : numpy.ndarray
            Atom indices of acceptors, donors, and acceptor-donors respectively.

        Notes
        -----
        Also populates ``self.don_H_pair_dict`` (donor atom -> list of
        [donor, bonded-hydrogen] pairs) and ``self.prot_hb_types`` (0 none,
        1 acceptor, 2 donor, 3 both), matching the SSTMap scheme.
        """
        top = self.topology
        try:
            top.create_standard_bonds()
        except Exception:  # noqa: BLE001 - bonds may already exist / template missing
            pass

        acc_list: list[int] = []
        don_list: list[int] = []
        acc_don_list: list[int] = []
        non_water_bonds = [
            (b[0].index, b[1].index)
            for b in top.bonds
            if b[0].residue.name not in WATER_RESNAMES
        ]
        dist_pairs: list[list[int]] = []

        for at in self.prot_atom_ids:
            if top.atom(at).element.name not in DON_ACC_ELEMENTS:
                continue
            bonds_of_at = [b for b in non_water_bonds if at in b]
            don_h_pairs = []
            for at1, at2 in bonds_of_at:
                if top.atom(at2).element.name == "hydrogen":
                    don_h_pairs.append([at1, at2])
                if top.atom(at1).element.name == "hydrogen":
                    don_h_pairs.append([at2, at1])

            element = top.atom(at).element.name
            if element == "nitrogen":
                if don_h_pairs:
                    dist_pairs.extend(don_h_pairs)
                    if at not in don_list:
                        don_list.append(at)
                else:
                    acc_list.append(at)
            else:  # oxygen or sulfur
                if don_h_pairs:
                    dist_pairs.extend(don_h_pairs)
                    if at not in acc_don_list:
                        acc_don_list.append(at)
                else:
                    acc_list.append(at)

        for pair in dist_pairs:
            self.don_H_pair_dict.setdefault(pair[0], []).append([pair[0], pair[1]])

        solute_acc_ids = np.array(acc_list, dtype=np.int64)
        solute_acc_don_ids = np.array(acc_don_list, dtype=np.int64)
        solute_don_ids = np.array(don_list, dtype=np.int64)

        for at_id in solute_acc_ids:
            self.prot_hb_types[at_id] = 1
        for at_id in solute_don_ids:
            self.prot_hb_types[at_id] = 2
        for at_id in solute_acc_don_ids:
            self.prot_hb_types[at_id] = 3
        return solute_acc_ids, solute_don_ids, solute_acc_don_ids

    def calculate_hydrogen_bonds(
        self, traj: md.Trajectory, water: int, nbrs: np.ndarray, water_water: bool = True
    ) -> np.ndarray:
        """Return the hydrogen bonds a water makes with its first-shell neighbours.

        A hydrogen bond is recorded when a donor-H...acceptor angle is within
        ``ANGLE_CUTOFF_RAD`` (30 degrees).

        Parameters
        ----------
        traj : mdtraj.Trajectory
            Single-frame trajectory used for the angle calculation.
        water : int
            Atom index of the query water's oxygen.
        nbrs : numpy.ndarray
            Oxygen atom indices (water-water) or solute atom indices
            (solute-water) of the first-shell neighbours.
        water_water : bool, optional
            Whether the neighbours are waters (default) or solute atoms.

        Returns
        -------
        numpy.ndarray, shape (n_hbonds, 3)
            Each row is a triplet ``[donor, acceptor-or-water, hydrogen]`` as
            produced by the underlying angle triplets.
        """
        angle_triplets = []
        if water_water:
            for wat_nbr in nbrs:
                angle_triplets.extend(
                    [
                        [water, wat_nbr, wat_nbr + 1],
                        [water, wat_nbr, wat_nbr + 2],
                        [wat_nbr, water, water + 1],
                        [wat_nbr, water, water + 2],
                    ]
                )
        else:
            for solute_nbr in nbrs:
                hb_type = self.prot_hb_types[solute_nbr]
                if hb_type in (1, 3):
                    angle_triplets.extend(
                        [[solute_nbr, water, water + 1], [solute_nbr, water, water + 2]]
                    )
                if hb_type in (2, 3):
                    for don_H_pair in self.don_H_pair_dict[solute_nbr]:
                        angle_triplets.extend([[water, solute_nbr, don_H_pair[1]]])
        if not angle_triplets:
            return np.empty((0, 3), dtype=np.int64)
        angle_triplets = np.asarray(angle_triplets)
        angles = md.compute_angles(traj, angle_triplets)
        angles[np.isnan(angles)] = 0.0
        return angle_triplets[np.where(angles[0, :] <= ANGLE_CUTOFF_RAD)]
