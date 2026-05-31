
from rdkit import Chem


def _process_inchi_options(
    fixed_hydrogen_layer: bool = True,
    undefined_stereocenter: bool = True,
    reconnected_metal_layer: bool = True,
    tautomerism_keto_enol: bool = True,
    tautomerism_15: bool = True,
    options: list[str] | None = None,
):
    inchi_options = []

    if fixed_hydrogen_layer:
        inchi_options.append("/FixedH")

    if undefined_stereocenter:
        inchi_options.append("/SUU")

    if reconnected_metal_layer:
        inchi_options.append("/RecMet")

    if tautomerism_keto_enol:
        inchi_options.append("/KET")

    if tautomerism_15:
        inchi_options.append("/15T")

    if options is not None:
        inchi_options.extend(options)

    inchi_options = " ".join(inchi_options)
    return inchi_options
    
    
def to_inchikey_non_standard(
    mol: Chem.Mol, 
    fixed_hydrogen_layer: bool = True,
    undefined_stereocenter: bool = True,
    reconnected_metal_layer: bool = True,
    tautomerism_keto_enol: bool = True,
    tautomerism_15: bool = True,
    options: list[str] | None = None,
) -> str | None:
    """Convert a mol to a non-standard InchiKey.

    Note that turning all the flags to `False` will result in the standard InchiKey.

    **Warning**: this function will return a **non-standard** InchiKey. See
    https://www.inchi-trust.org/technical-faq-2 for details.

    It's important to not mix standard and non-standard InChiKey. If you don't know
    much about non-standard InchiKey, we highly recommend you to use the
    standard InchiKey with `dm.to_inchikey()`.

    Args:
        mol: a molecule
        fixed_hydrogen_layer: whether to include a fixed hydrogen layer (`/FixedH`).
        undefined_stereocenter: whether to include an undefined stereocenter layer (`/SUU`).
        reconnected_metal_layer: whether to include reconnected metals (`/RecMet`).
        tautomerism_keto_enol: whether to account tautomerism keto-enol (`/KET`).
        tautomerism_15: whether to account 1,5-tautomerism (`/15T`).
        options: More InchI options in a form of a list of string. Example:
            `["/SRel", "/AuxNone"]`.
    """

    if mol is None:
        return None

    inchi_options = _process_inchi_options(
        fixed_hydrogen_layer=fixed_hydrogen_layer,
        undefined_stereocenter=undefined_stereocenter,
        reconnected_metal_layer=reconnected_metal_layer,
        tautomerism_keto_enol=tautomerism_keto_enol,
        tautomerism_15=tautomerism_15,
        options=options,
    )

    inchikey = Chem.MolToInchiKey(mol, options=inchi_options)
    if not inchikey:
        return None
    return inchikey