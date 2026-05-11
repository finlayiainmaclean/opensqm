# pyrefly: ignore [missing-import]
from openmm.unit import kilojoules_per_mole, kelvin, is_quantity, MOLAR_GAS_CONSTANT_R
from scipy.optimize import curve_fit
import numpy as np
from collections import defaultdict
from openmm.app import element as elem
from opensqm.cph.constantph import ConstantPH
from pathlib import Path
from openmm.app import PDBFile
import joblib
from loguru import logger

from opensqm.cph.simulation_config import SimulationConfig

MODEL_COMPOUNDS = [
    # ('CYS.pdb', ['CYS', 'CYX'], 8.33),
    ('HIS.pdb', ['HIP', 'HID'], 7.1),
    ('HIS.pdb', ['HIP', 'HIE'], 6.5),
    # ('ASP.pdb', ['ASH', 'ASP'], 3.7),
    # ('GLU.pdb', ['GLH', 'GLU'], 4.3),
    # ('LYS.pdb', ['LYS', 'LYN'], 10.4),
    ]


def get_hydrogen_variants(topology):
    """Given an OpenMM Topology, return a per-residue list of (hydrogen_name, parent_name) tuples.
    
    This produces output in the format expected by Modeller.addHydrogens(variants=...).
    """
    
    
    # Build per-residue hydrogen -> parent mapping from bonds
    residue_hydrogens = defaultdict(list)
    for bond in topology.bonds():
        a1, a2 = bond[0], bond[1]
        if a1.element == elem.hydrogen and a1.residue == a2.residue:
            residue_hydrogens[a1.residue.index].append((a1.name, a2.name))
        elif a2.element == elem.hydrogen and a1.residue == a2.residue:
            residue_hydrogens[a2.residue.index].append((a2.name, a1.name))
    
    # Build the variants list (one entry per residue)
    variants: list[list[tuple[str, str]] | None] = []
    for residue in topology.residues():
        if residue.index in residue_hydrogens:
            variants.append(residue_hydrogens[residue.index])
        else:
            variants.append(None)
    return variants



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

    def findReferenceEnergies(self, iterations=20000, substeps=20):
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
                self.model.attemptMCStep(self.temperature)
            fractions = [[] for _ in range(len(self.model.pH))]
            for i in range(iterations):
                self.model.simulation.step(substeps)
                self.model.attemptMCStep(self.temperature)
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
            popt, pcov = curve_fit(f, x, y, [0.0], full_output=False) 
            root = popt[0]
            if root > -2 and root < 2:
                self.titration.referenceEnergies[1] += scale*(self.pKa-root)
                break
            self.titration.referenceEnergies[1] -= scale*root
    

def generate_all(config: SimulationConfig, iterations=20000, substeps=20):
    """Generate and cache reference energies for all model compounds.

    Parameters
    ----------
    config: SimulationConfig
        The simulation configuration to use.
    iterations: int
        The number of Monte Carlo moves to attempt per compound.
    substeps: int
        The number of dynamics steps to integrate between Monte Carlo moves.

    Returns
    -------
    dict
        Mapping of (variant1, variant2) tuples to reference energy tuples.
    """

    model_compound_dir = Path(__file__).resolve().parent / "model-compounds"
    reference_energies_dir = model_compound_dir / "reference_energies"
    reference_energies_dir.mkdir(exist_ok=True)

    reference_energies = {}

    for pdb_name, variants, model_pka in MODEL_COMPOUNDS:
        cache_name = "-".join(variants)
        cache_path = reference_energies_dir / f"{cache_name}_{config.hash()}.pkl"

        if cache_path.exists():
            logger.info(f"Skipping {cache_name}: cached at {cache_path}")
            reference_energies[tuple(variants)] = joblib.load(cache_path)
            continue

        pdb = PDBFile(str(model_compound_dir / pdb_name))
        variants_dict = {1: variants}

        cph = ConstantPH(topology=pdb.topology,
                positions=pdb.positions,
                pH=7.0,
                config=config,
                residueVariants=variants_dict,
                referenceEnergies={1: [0.0, 0.0]})

        finder = ReferenceEnergyFinder(cph, pKa=model_pka, temperature=config.temperature)
        finder.findReferenceEnergies(iterations=iterations, substeps=substeps)
        ref_energies = cph.titrations[1].referenceEnergies
        logger.info(f"Computed reference energies for {cache_name}: {ref_energies}")
        ref_energies = tuple([v._value for v in ref_energies])

        joblib.dump(ref_energies, cache_path)
        logger.info(f"Saved {cache_name} -> {cache_path}")
        reference_energies[tuple(variants)] = ref_energies

    return reference_energies


if __name__=="__main__":
    

    config = SimulationConfig()
    logger.info(f"Hash: {config.hash()}")
    reference_energies = generate_all(config)
    logger.info(f"Reference energies: {reference_energies}")