from openmm import *
from openmm.app import *
from openmm import unit

def get_params(pH = 7.4, temperature=300):
    explicit_ff = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    implicit_ff = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
    explicit_params = dict(nonbondedMethod=PME, nonbondedCutoff=0.9*unit.nanometers, constraints=HBonds, hydrogenMass=1.5*unit.amu)
    implicit_params = dict(nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=2.0*unit.nanometers, constraints=HBonds)
    temperature = 300*unit.kelvin
    integrator = LangevinIntegrator(temperature, 1.0/unit.picosecond, 0.004*unit.picoseconds)
    relaxation_integrator = LangevinIntegrator(temperature, 10.0/unit.picosecond, 0.002*unit.picosecond)
    return explicit_ff, implicit_ff, explicit_params, implicit_params, integrator, relaxation_integrator