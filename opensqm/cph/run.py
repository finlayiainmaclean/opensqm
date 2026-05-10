# pyrefly: ignore [missing-import]
from openmm import *
# pyrefly: ignore [missing-import]
from openmm.app import *
# pyrefly: ignore [missing-import]
from openmm import unit
from opensqm.cph.constantph import ConstantPH
from opensqm.cph.reference_energy import ReferenceEnergyFinder
import numpy as np
import joblib
from opensqm.cph.simulation_config import SimulationConfig
from opensqm.md.fix import run_pdbfixer
# pyrefly: ignore [missing-import]
from opensqm.cph.generate_reference_energies import MODEL_PKAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load your data (you probably already have this)
# df = pd.read_csv('ph_states.csv')

def henderson_hasselbalch(pH, pKa):
    return 1 / (1 + 10**(pH - pKa))


def calculate_pkas(df, reference_pkas):
    pkas = {}
    for residue_index, reference_pka in reference_pkas.items():
        frac = (
            df.groupby('ph')[residue_index]
            .apply(lambda x: np.mean(x == 0))
            .reset_index(name='f_prot')
        )

        # Fit curve
        popt, pcov = curve_fit(henderson_hasselbalch, frac['ph'], frac['f_prot'], p0=[reference_pka])
        pka = popt[0]
        perr = np.sqrt(np.diag(pcov))
        pka, pka_err = popt[0], perr[0]
        pkas[residue_index] = (pka, pka_err)
    return pkas





config = SimulationConfig()

################################################################

fixer = run_pdbfixer(input_protein_path="1AKI.pdb",
             output_protein_path="/tmp/fixed.pdb")

# equilibrate(
#     input_protein_path="/tmp/fixed.pdb",
#     output_complex_path="/tmp/equilibrated.pdb",
#     npt_ps=1000,
#     nvt_ps=1000,
#     warmup_ps=100)


################################################################

# specific_residue_ids = [("A", "HIS", 15)]
specific_residue_ids = []

pdb = PDBFile("/tmp/equilibrated.pdb")

residues = {
#  'CYS': ['CYS', 'CYX'],
 "HIS": ['HIP', 'HID', 'HIE'],
 "ASP": ['ASH', 'ASP'],
 "GLU": ['GLH', 'GLU'],
#  "LYS": ['LYS', 'LYN']
}

variants = {}
reference_energies = {}
reference_pkas = {}
for residue in pdb.topology.residues():
    if residue.name in residues:
        residue_id = (residue.chain.id, residue.name, int(residue.id))

        if residue_id in specific_residue_ids or len(specific_residue_ids)==0:
            print(f"Adding {residue_id}")
            variants[residue.index] = residues[residue.name]
            reference_energies[residue.index] = config.reference_energies[tuple(residues[residue.name])]
            reference_pkas[residue.index] = MODEL_PKAS[residue.name]


################################################################

min_pH = -3
max_pH = 8
pH_spacing = 1.0
temperature = 300 * unit.kelvin
pHs = np.arange(min_pH, max_pH + pH_spacing, pH_spacing)
pHs = [float(pH) for pH in pHs]
print(pHs)
cph = ConstantPH(pdb.topology, pdb.positions, pHs, config, variants, reference_energies)
cph.simulation.minimizeEnergy()
cur_weights = cph.weights

num_successful_swaps = 0
for i in range(10000):
    prev_weights = np.array(cur_weights)
    cph.simulation.step(50)
    swap = cph.attemptMCStep(temperature)
    num_successful_swaps += int(swap)
    cur_weights = np.array(cph.weights)
    diff = np.linalg.norm(cur_weights - prev_weights)
    
    if i % 100 == 0:
        print(diff)
    if diff<1e-1:
        print("Converged")
        break
else:
    print("Did not converge")
print("num_successful_swaps", num_successful_swaps)
    

################################################################ 
from tqdm import tqdm
import pandas as pd

run_time = 120 * 1000 # 1ns
num_steps = int(run_time/0.2)

results = []

for i in tqdm(range(num_steps)):
    cph.simulation.step(50) # 0.2ps
    cph.attemptMCStep(temperature)
    result = (cph.pH[cph.currentPHIndex], *[cph.titrations[index].currentIndex for index in variants])
    results.append(result)
    

    if i>0 and i % 1000 == 0:
        df = pd.DataFrame(results, columns = ['ph', *variants.keys()])
        df.to_csv("results.csv", index=False)
        print(calculate_pkas(df, reference_pkas))





