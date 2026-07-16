"""Static residue-level configuration shared by the model-compound pipeline.

Two pieces live together here because they both describe the *protein*
side of the titratable-residue catalogue:

* :data:`_PROTEIN_VARIANT_CHARGES` is a flat name -> formal-charge lookup
  used by the legacy-cache migration in
  :class:`opensqm.cph.reference_energy.models.TitratableResidueReference`.
  It is the single source of truth for protein variant charges; new
  variants (e.g. an extra HIN imidazolate) must be registered here
  *and* in :data:`MODEL_COMPOUNDS` before the rest of the package can
  use them.
* :data:`MODEL_COMPOUNDS` describes each model compound's PDB file,
  variant list, formal charges and microscopic transitions. It is
  consumed by :func:`opensqm.cph.reference_energy.generate.generate_all`
  to compute (or load) reference energies once per
  :class:`opensqm.cph.simulation_config.SimulationConfig` hash.
"""

# Formal-charge lookup for protein variants used by ``MODEL_COMPOUNDS`` and
# by the legacy-cache migration path. Add new entries here when introducing
# new protonation states (e.g. ``HIN`` for the imidazolate).
_PROTEIN_VARIANT_CHARGES: dict[str, int] = {
    "HIP": 1,
    "HID": 0,
    "HIE": 0,
    "HIN": -1,
    "CYS": 0,
    "CYX": -1,
    "ASH": 0,
    "ASP": -1,
    "GLH": 0,
    "GLU": -1,
    "LYS": 1,
    "LYN": 0,
    # ASN/GLN are not titratable but participate in ring-flip MC
    # (the classic MolProbity O<->N flip about CB-CG / CG-CD); registered
    # here as single-variant entries so ConstantPH picks them up.
    "ASN": 0,
    "GLN": 0,
}


MODEL_COMPOUNDS: dict[str, dict] = {
    "CYS": {
        "pdb_name": "CYS.pdb",
        "main_variant": "CYS",
        "variants": ["CYS", "CYX"],
        "charges": [0, -1],
        "transitions": [("CYS", "CYX", 8.33)],
    },
    "HIS": {
        "pdb_name": "HIS.pdb",
        "main_variant": "HIP",
        "variants": ["HIP", "HID", "HIE"],
        "charges": [1, 0, 0],
        "transitions": [
            ("HIP", "HID", 7.1),
            ("HIP", "HIE", 6.5),
        ],
        # 180-deg flip of the imidazole ring about CB-CG; useful MC move
        # because the two ring nitrogens hydrogen-bond very differently and
        # plain MD has a hard time crossing the barrier on simulation
        # timescales.
        "ring_flip_bonds": [("CB", "CG")],
    },
    "ASP": {
        "pdb_name": "ASP.pdb",
        "main_variant": "ASH",
        "variants": ["ASH", "ASP"],
        "charges": [0, -1],
        "transitions": [("ASH", "ASP", 3.7)],
    },
    "GLU": {
        "pdb_name": "GLU.pdb",
        "main_variant": "GLH",
        "variants": ["GLH", "GLU"],
        "charges": [0, -1],
        "transitions": [("GLH", "GLU", 4.3)],
    },
    "LYS": {
        "pdb_name": "LYS.pdb",
        "main_variant": "LYS",
        "variants": ["LYS", "LYN"],
        "charges": [1, 0],
        "transitions": [("LYS", "LYN", 10.4)],
    },
    # ASN / GLN: single-variant entries that exist purely to register a
    # ring-flip MC move on the side-chain amide. The 180-deg flip about
    # CB-CG (ASN) / CG-CD (GLN) swaps the carbonyl O with the amide N -
    # the classic MolProbity flip. With only one variant there is no
    # protonation transition to attempt and ``transitions`` stays empty;
    # ConstantPH skips the protonation MC for these residues and uses
    # _attemptRingFlip exclusively. The ``pdb_name`` field is unused
    # (the model-compound pairwise solver only reads it inside the
    # ``for t in ordered_transitions`` loop, which is empty here).
    "ASN": {
        "pdb_name": "ASN.pdb",
        "main_variant": "ASN",
        "variants": ["ASN"],
        "charges": [0],
        "transitions": [],
        "ring_flip_bonds": [("CB", "CG")],
    },
    "GLN": {
        "pdb_name": "GLN.pdb",
        "main_variant": "GLN",
        "variants": ["GLN"],
        "charges": [0],
        "transitions": [],
        "ring_flip_bonds": [("CG", "CD")],
    },
}


__all__ = ["MODEL_COMPOUNDS", "_PROTEIN_VARIANT_CHARGES"]
