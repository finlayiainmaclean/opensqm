"""Pairwise reference-energy fitting via simulated-tempering MC.

This module hosts the iterative fitting machinery used to translate a
single experimental microscopic pKa into a per-state reference-energy
offset. There are two entry points:

* :class:`ReferenceEnergyFinder` -- the underlying object that, given a
  :class:`opensqm.cph.constantph.ConstantPH` model with exactly two
  states and a target pKa, refines ``referenceEnergies[1]`` until the
  protonation fraction crosses 0.5 at the requested pKa.
* :func:`_compute_pairwise_reference_energy` -- a thin wrapper that
  spins up a fresh ``ConstantPH`` from a model-compound PDB for a
  ``(parent, child)`` variant pair and returns the resulting kJ/mol
  offset of the child relative to the parent. Reserved for the
  ``MODEL_COMPOUNDS`` loop in
  :func:`opensqm.cph.reference_energy.generate.generate_all`; the
  ligand path inlines an equivalent block so it can re-use a single
  solvated topology across all transitions.
"""
# pyrefly: ignore [missing-import]
from pathlib import Path

import numpy as np
from loguru import logger
from openmm.app import PDBFile
from openmm.unit import (
    MOLAR_GAS_CONSTANT_R,
    is_quantity,
    kelvin,
    kilojoules_per_mole,
)
from scipy.optimize import curve_fit

from opensqm.cph.constantph import ConstantPH
from opensqm.cph.simulation_config import ConstantpHSettings

from .models import TitratableResidueReference, Transition
from .types import VariantSpec


class ReferenceEnergyFinder(object):
    def __init__(self, model, pKa, temperature):
        """
        Construct a ReferenceEnergyFinder.

        Parameters
        ----------
        model: ConstantPH
            The model for which to determine reference energies.  It must contain a single titratable residue with
            exactly two states.  It does not matter what pH or reference energies were specified when it was created,
            because they will both be overwritten.
        pKa: float
            The experimental pKa of the titratable residue.  Reference energies will be chosen to match it.
        temperature: openmm.unit.Quantity
            The temperature at which the simulation will be run.
        """
        if len(model.titrations) != 1:
            raise ValueError("The model compound must contain a single titratable residue")
        self.model = model
        self.pKa = pKa
        if not is_quantity(temperature):
            temperature = temperature*kelvin
        self.temperature = temperature
        self.residueIndex = list(model.titrations.keys())[0]
        self.titration = model.titrations[self.residueIndex]
        if len(self.titration.explicitStates) != 2:
            raise ValueError("Only residues with two states are currently supported")

    def findReferenceEnergies(self, iterations=10, substeps=20):
        """
        Compute the reference energies for the states of the model compound.  On exit, they will be stored in
        the ConstantPH object.

        Parameters
        ----------
        iterations: int
            The number of Monte Carlo moves to attempt.  The larger the number, the more tightly converged
            the results will be.
        subsets: int
            The number of dynamics steps to integrate between Monte Carlo moves.
        """
        # Find an initial estimate of the reference energies just by computing the potential
        # energies of the states.

        self.model.setResidueState(self.residueIndex, 0)
        energy0 = self.model.implicitContext.getState(energy=True).getPotentialEnergy()
        self.model.setResidueState(self.residueIndex, 1)
        energy1 = self.model.implicitContext.getState(energy=True).getPotentialEnergy()
        deltaN = self.titration.implicitStates[1].numHydrogens - self.titration.implicitStates[0].numHydrogens
        scale = MOLAR_GAS_CONSTANT_R*self.temperature*deltaN*np.log(10.0)
        self.titration.referenceEnergies[0] = 0.0*kilojoules_per_mole
        self.titration.referenceEnergies[1] = energy1-energy0
        self.model.simulation.minimizeEnergy()
        self.model.simulation.context.setVelocitiesToTemperature(self.temperature)

        # If our initial estimate is exact, the fractions should be equal at pH 0.  Since it probably
        # isn't, simulate it at various pHs to refine the estimate.

        while True:
            self.model.setPH([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
            for i in range(1000):
                self.model.simulation.step(substeps)
                self.model.attemptMCStep()
            fractions = [[] for _ in range(len(self.model.pH))]
            for i in range(iterations):
                self.model.simulation.step(substeps)
                self.model.attemptMCStep()
                fractions[self.model.currentPHIndex].append(1.0 if self.titration.protonatedIndex == self.titration.currentIndex else 0.0)

            # Fit a curve to the data to better estimate when the fraction is exactly 0.5,
            # and compute the reference energy based on it.

            x = []
            y = []
            for i in range(len(fractions)):
                if len(fractions[i]) > 0:
                    x.append(self.model.pH[i])
                    y.append(np.average(fractions[i]))

            def f(ph, pka):
                return 1/(1+10**(ph-pka))

            # pyrefly: ignore [bad-unpacking]
            popt, _pcov = curve_fit(f, x, y, [0.0], full_output=False)
            root = popt[0]
            if root > -2 and root < 2:
                self.titration.referenceEnergies[1] += scale*(self.pKa-root)
                break
            self.titration.referenceEnergies[1] -= scale*root


def _make_pair_reference(
    residue_name: str,
    pair_variants: list[VariantSpec],
    pair_names: list[str],
    pair_charges: list[int],
    pka: float,
) -> TitratableResidueReference:
    """Build a 2-state ``TitratableResidueReference`` for a pairwise fit.

    The reference energies are seeded to ``[0.0, 0.0]`` so the finder can
    refine the child energy in place via ``ResidueTitration.referenceEnergies``
    (which is a defensive local copy of these). The single transition
    ``0 -> 1`` is supplied at the requested pKa.
    """
    return TitratableResidueReference(
        residue_name=residue_name,
        main_variant=pair_names[0],
        variant_names=pair_names,
        variants=pair_variants,
        charges=pair_charges,
        reference_energies_kj_per_mole=[0.0, 0.0],
        transitions=[Transition(parent=0, child=1, pka=float(pka))],
    )


def _compute_pairwise_reference_energy(
    model_compound_dir: Path,
    pdb_name: str,
    residue_name: str,
    pair: list[VariantSpec],
    pair_charges: list[int],
    pka: float,
    config: ConstantpHSettings,
    iterations: int,
    substeps: int,
) -> float:
    """Return the reference energy (kJ/mol) of ``pair[1]`` relative to ``pair[0]``.

    The two entries of ``pair`` are passed straight through to ``ConstantPH``
    via a synthetic 2-state :class:`TitratableResidueReference`. ``pka`` is
    the experimental micro pKa of the parent->child deprotonation, where
    ``parent`` is the more-protonated state.
    """
    if any(isinstance(p, str) for p in pair):
        # For protein variants the variant name *is* the string entry.
        pair_names = [p if isinstance(p, str) else f"<custom_{i}>"
                      for i, p in enumerate(pair)]
    else:
        pair_names = [f"{residue_name}_{i}" for i in range(len(pair))]
    cache_name = "-".join(pair_names)

    pdb = PDBFile(str(model_compound_dir / pdb_name))
    # The model-compound PDBs always carry the residue under index 1 (atom 0
    # is the leading ACE cap, atom 1 is the residue itself), so re-use that.
    pair_reference = _make_pair_reference(
        residue_name=residue_name,
        pair_variants=list(pair),
        pair_names=pair_names,
        pair_charges=list(pair_charges),
        pka=pka,
    )
    # ConstantPH keys references by topology residue name. Model-compound
    # PDBs use the protein residue name (e.g. ``"HIS"``) so we register the
    # synthetic pair under that name; the variable residue sits at index 1
    # between the ACE/NME caps.
    pdb_residues = list(pdb.topology.residues())
    references = {pdb_residues[1].name: pair_reference}
    cph = ConstantPH(
        topology=pdb.topology,
        positions=pdb.positions,
        pH=7.0,
        config=config,
        references=references,
        titratable_residue_indices=[1],
        ring_flip_angles=None,
    )
    logger.info(f"Computing reference energy for {cache_name} with pKa={pka}")
    finder = ReferenceEnergyFinder(cph, pKa=pka, temperature=config.temperature)
    finder.findReferenceEnergies(iterations=iterations, substeps=substeps)
    ref_energies = cph.titrations[1].referenceEnergies
    logger.info(f"Computed reference energies for {cache_name}: {ref_energies}")
    # ``referenceEnergies`` is in the order of the input ``pair``. ``pair[0]``
    # is the parent slot supplied by the caller, so ``ref_energies[1]`` is
    # the energy of ``pair[1]`` relative to ``pair[0]``.
    return float(ref_energies[1]._value)


__all__ = ["ReferenceEnergyFinder", "_compute_pairwise_reference_energy"]
