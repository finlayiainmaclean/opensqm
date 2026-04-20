from openmm.app import ForceField

ff = ForceField("amber/ff14SB.xml", "amber/tip3p_standard.xml")
print("Before:", len(ff._forces))

ff.loadFile("implicit/obc2.xml")
print("After loading OBC2:", len(ff._forces))
