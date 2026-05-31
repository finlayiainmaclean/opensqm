"""Build aligned RDKit Mols for a list of ligand protonation states.

The public API here is :func:`build_protonation_template` (returns the
maximally-protonated "super-template") and :func:`build_protonation_states`
(returns one Mol per requested protomer, all sharing heavy-atom indices,
PDB-style atom names and conformer geometry with the template). Together
they produce inputs that ``Modeller.addHydrogens(variants=...)`` can use
to flip a custom ligand between its protonation states without having
to re-embed or re-name atoms each time.

The remaining helpers are private support routines used to canonicalise
heavy-atom skeletons, locate the proton-loss differences between a
template and a target protomer, and re-shape templates accordingly.
"""
# pyrefly: ignore [missing-import]
from rdkit import Chem
from rdkit.Chem import AllChem


def _assign_ligand_atom_names(rdkit_mol, residue_name="LIG"):
    """Assign unique PDB-style atom names and residue info to an rdkit Mol in place.

    Heavy atoms get names like 'N1', 'C2', ...; hydrogens get 'H1', 'H2', ...
    `Modeller.addHydrogens` keys variants off parent atom names, so it is
    important that the heavy-atom naming is consistent across all states of
    a titratable ligand.
    """
    elem_count = {}
    for atom in rdkit_mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        sym = atom.GetSymbol()
        elem_count[sym] = elem_count.get(sym, 0) + 1
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f"{sym}{elem_count[sym]:<3}")
        info.SetResidueName(residue_name)
        info.SetResidueNumber(1)
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)

    h_count = 0
    for atom in rdkit_mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        h_count += 1
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f"H{h_count:<3}")
        info.SetResidueName(residue_name)
        info.SetResidueNumber(1)
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)


def _heavy_skeleton(mol: Chem.Mol) -> Chem.Mol:
    """Return ``mol``'s heavy-atom skeleton with all formal charges and explicit
    H counts zeroed.

    Used as a canonical structure for substructure matching when we want to
    compare two protonation states of the same chemistry without the
    matching being sensitive to per-atom charge or H count.
    """
    skel = Chem.RWMol(Chem.RemoveHs(mol))
    for atom in skel.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    return skel.GetMol()


def _heavy_features(mol: Chem.Mol) -> tuple[list[int], list[int]]:
    """Return per-heavy-atom ``(num_explicit_Hs, formal_charge)`` lists."""
    h_counts: list[int] = []
    charges: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        h_counts.append(
            sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
        )
        charges.append(atom.GetFormalCharge())
    return h_counts, charges


def _find_proton_loss_diff(
    template: Chem.Mol, target: Chem.Mol,
) -> list[int]:
    """Return template heavy-atom indices to deprotonate to reach ``target``.

    Both inputs must have all hydrogens explicit. The matching is done on
    the bare heavy-atom skeleton (formal charges and H counts ignored),
    then we pick a heavy-atom isomorphism in which, atom-for-atom, the
    drop in attached-H count equals the drop in formal charge -- the
    invariant of losing a proton. Pinning both quantities together
    disambiguates matches that the skeleton alone leaves degenerate (e.g.
    imidazole's two ring nitrogens look identical to imidazolium's after
    charges are stripped). Returned indices repeat when multiple Hs are
    lost from the same site.

    Raises ``ValueError`` if the two skeletons are not isomorphic, or if
    ``target`` is more protonated than ``template`` at any heavy atom.
    """
    template_skel = _heavy_skeleton(template)
    target_skel = _heavy_skeleton(target)
    matches = template_skel.GetSubstructMatches(
        target_skel, useChirality=False, uniquify=False,
    )
    if not matches:
        raise ValueError(
            "Template and target do not share a heavy-atom skeleton"
        )

    template_hs, template_qs = _heavy_features(template)
    target_hs, target_qs = _heavy_features(target)

    for match in matches:
        # match[t] is the template heavy-atom index corresponding to
        # target heavy atom t.
        h_diffs = [
            template_hs[match[t]] - target_hs[t] for t in range(len(match))
        ]
        q_diffs = [
            template_qs[match[t]] - target_qs[t] for t in range(len(match))
        ]
        if (
            all(d >= 0 for d in h_diffs)
            and all(d >= 0 for d in q_diffs)
            and h_diffs == q_diffs
        ):
            sites: list[int] = []
            for t_idx, n_drop in enumerate(h_diffs):
                sites.extend([match[t_idx]] * n_drop)
            return sites

    raise ValueError(
        "Could not find a valid proton-loss isomorphism (target may be more "
        "protonated than template, or the skeletons may differ)"
    )


def _derive_state_from_template(
    template: Chem.Mol, target: Chem.Mol,
) -> Chem.Mol:
    """Return a copy of ``template`` deprotonated to match ``target``.

    Heavy atoms keep their indices, names, and conformer coordinates from
    the template; only the hydrogens determined by
    :func:`_find_proton_loss_diff` are removed, with their parent heavy
    atoms' formal charges adjusted accordingly. Returns the template
    unchanged when no protons need to be lost.
    """
    sites = _find_proton_loss_diff(template, target)
    rwmol = Chem.RWMol(template)
    if not sites:
        return rwmol.GetMol()

    # Resolve which explicit H atom to remove for each site up-front: we
    # walk the template once, pop one H neighbour per requested removal,
    # then delete from highest to lowest index so earlier removals don't
    # invalidate later ones.
    remaining_hs: dict[int, list[int]] = {}
    for atom in template.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        remaining_hs[atom.GetIdx()] = [
            nbr.GetIdx() for nbr in atom.GetNeighbors()
            if nbr.GetAtomicNum() == 1
        ]

    h_indices_to_remove: list[int] = []
    for site in sites:
        if not remaining_hs[site]:
            raise ValueError(
                f"Template atom {site} has no remaining hydrogens to remove"
            )
        h_indices_to_remove.append(remaining_hs[site].pop())

    for h_idx in sorted(h_indices_to_remove, reverse=True):
        rwmol.RemoveAtom(h_idx)

    site_drop_counts: dict[int, int] = {}
    for s in sites:
        site_drop_counts[s] = site_drop_counts.get(s, 0) + 1
    for site, n_drop in site_drop_counts.items():
        atom = rwmol.GetAtomWithIdx(site)
        atom.SetFormalCharge(atom.GetFormalCharge() - n_drop)

    Chem.SanitizeMol(rwmol)
    return rwmol.GetMol()


def _ensure_explicit_hs(state: Chem.Mol | str) -> Chem.Mol:
    """Coerce a SMILES string or RDKit Mol to a Mol with explicit hydrogen atoms.

    Always returns a fresh ``Chem.Mol`` so subsequent edits don't mutate
    the caller's input.
    """
    if isinstance(state, str):
        return Chem.AddHs(Chem.MolFromSmiles(state))
    has_explicit_h = any(atom.GetAtomicNum() == 1 for atom in state.GetAtoms())
    return Chem.Mol(state) if has_explicit_h else Chem.AddHs(state)


def _map_states_to_canonical_skeleton(
    states_with_hs: list[Chem.Mol],
) -> tuple[Chem.Mol, list[tuple[int, ...]]]:
    """Return a canonical heavy-atom skeleton and per-state index mappings.

    The skeleton is taken from ``states_with_hs[0]`` after stripping all
    Hs, formal charges, and explicit-H counts. ``mappings[i][k]`` is the
    canonical heavy-atom index for ``states_with_hs[i]``'s heavy atom k.
    Raises ``ValueError`` if any state's skeleton is not isomorphic to the
    canonical one.
    """
    canonical = _heavy_skeleton(states_with_hs[0])
    n_heavy = canonical.GetNumAtoms()
    mappings: list[tuple[int, ...]] = []
    for state in states_with_hs:
        skel = _heavy_skeleton(state)
        match = canonical.GetSubstructMatch(skel)
        if not match or len(match) != n_heavy:
            raise ValueError(
                "All protonation states must share the same heavy-atom "
                "skeleton; got mismatched structures"
            )
        mappings.append(match)
    return canonical, mappings


def build_protonation_template(
    states: list[Chem.Mol] | list[str],
    seed: int = 42,
) -> Chem.Mol:
    """Build a maximally-protonated "super-template" from a list of protomers.

    For every heavy atom in the shared skeleton, the returned Mol carries
    the maximum H count *and* the maximum formal charge observed across
    the inputs -- i.e. the molecule with every titration site
    simultaneously protonated, even when no single input state has all
    sites protonated. For example, given

        - H3N+-CH2-CH2-NH2  (site 1 protonated)
        - H2N-CH2-CH2-NH3+  (site 2 protonated)
        - H2N-CH2-CH2-NH2   (both deprotonated)

    this returns the di-cation H3N+-CH2-CH2-NH3+ that none of the inputs
    contain on its own. The template is embedded with ETKDG, UFF-optimized,
    and has PDB-style heavy-atom and H names assigned.

    All inputs must share a heavy-atom skeleton (i.e. only protonation
    states of the same chemistry, not different tautomers with shifted
    heavy atoms). Inputs may be SMILES strings or RDKit Mols and are
    accepted with or without explicit Hs.
    """
    if not states:
        raise ValueError("states must not be empty")

    states_with_hs = [_ensure_explicit_hs(s) for s in states]
    canonical_skel, mappings = _map_states_to_canonical_skeleton(states_with_hs)
    n_heavy = canonical_skel.GetNumAtoms()

    # Per-canonical-atom maxima. ``max_q`` is initialised None so we can
    # cope with the (legitimate) case where every state has a negative
    # formal charge at that atom.
    max_h: list[int] = [0] * n_heavy
    max_q: list[int | None] = [None] * n_heavy
    for state, mapping in zip(states_with_hs, mappings):
        h_counts, charges = _heavy_features(state)
        for state_idx in range(n_heavy):
            c_idx = mapping[state_idx]
            if h_counts[state_idx] > max_h[c_idx]:
                max_h[c_idx] = h_counts[state_idx]
            cur_q = max_q[c_idx]
            if cur_q is None or charges[state_idx] > cur_q:
                max_q[c_idx] = charges[state_idx]

    template_rw = Chem.RWMol(canonical_skel)
    for j in range(n_heavy):
        atom = template_rw.GetAtomWithIdx(j)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(max_h[j])
        q = max_q[j]
        atom.SetFormalCharge(0 if q is None else q)

    template = template_rw.GetMol()
    try:
        Chem.SanitizeMol(template)
    except Chem.AtomValenceException as exc:
        raise ValueError(
            f"Merged super-template is chemically invalid: cannot fit "
            f"H-counts {max_h} with formal charges {max_q} on the shared "
            f"heavy-atom skeleton"
        ) from exc

    template = Chem.AddHs(template)
    if AllChem.EmbedMolecule(template, randomSeed=seed) == -1:
        raise RuntimeError(
            "Failed to embed a 3D conformer for the merged super-template"
        )
    AllChem.UFFOptimizeMolecule(template)
    _assign_ligand_atom_names(template)
    return template


def build_protonation_states(
    states: list[Chem.Mol] | list[str],
    seed: int = 42,
) -> list[Chem.Mol]:
    """Build aligned RDKit Mols for a list of protonation states.

    Internally synthesises a maximally-protonated template via
    :func:`build_protonation_template` so that every titration site is
    represented even when no single input state has all sites protonated.
    Each input state is then derived from that template by locating which
    hydrogens differ via a heavy-atom substructure match and removing them.
    All returned mols share heavy-atom indices, PDB-style names, and
    conformer geometry -- the prerequisite for using them together with
    ``Modeller.addHydrogens(variants=...)``. The returned list preserves
    the input order, so callers can pass e.g. a unipka microstate
    distribution in whichever order is most convenient.
    """
    if not states:
        raise ValueError("states must not be empty")

    states_with_hs = [_ensure_explicit_hs(s) for s in states]
    template = build_protonation_template(states_with_hs, seed=seed)
    return [
        _derive_state_from_template(template, state)
        for state in states_with_hs
    ]


__all__ = ["build_protonation_template", "build_protonation_states"]
