from openff.toolkit.topology import Molecule
from openmm import GBSAOBCForce
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1OC"))
AllChem.EmbedMolecule(mol)

offmol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
from openff.toolkit.utils.toolkits import RDKitToolkitWrapper

offmol.assign_partial_charges("gasteiger", toolkit_registry=RDKitToolkitWrapper())

from openff.toolkit.typing.engines.smirnoff import ForceField

ff = ForceField("openff-2.2.0.offxml")
top = offmol.to_topology()

sys_ff = ff.create_openmm_system(top)

# map elements to radii (OBC2 mbondi2)
radii = {
    1: (0.12, 0.85),
    6: (0.17, 0.72),
    7: (0.155, 0.79),
    8: (0.15, 0.85),
    9: (0.15, 0.88),
    15: (0.185, 0.86),
    16: (0.18, 0.96),
    17: (0.17, 0.8),  # generic halogen
}

gbsa = GBSAOBCForce()
gbsa.setSolventDielectric(78.5)
gbsa.setSoluteDielectric(1.0)
for atom in offmol.atoms:
    atomic_number = atom.atomic_number
    charge = atom.partial_charge.m
    rad, scale = radii.get(atomic_number, (0.15, 0.8))  # fallback
    gbsa.addParticle(charge, rad, scale)

sys_ff.addForce(gbsa)

print(f"Added GBSA force with {gbsa.getNumParticles()} particles.")
print("Success! System created.")
