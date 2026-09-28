"""The MMGBSA protomer funnel on a ligand uniKa cannot enumerate."""

from rdkit import Chem
from rdkit.Chem import AllChem
from unipka.unipka import EnumerationError

from opensqm.md.run_mmgbsa import _enumerate_protomers


class _OneChargeState:
    """Stands in for UnipKa on a ligand with no ionisable site."""

    def get_distribution(self, mol: Chem.Mol, pH: float) -> None:  # noqa: N803
        raise EnumerationError("Failed to enumerate microstates across 2 charge states")


def test_a_ligand_with_one_charge_state_scores_its_input_protonation(tmp_path) -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("NC(=O)c1ccccc1"))  # benzamide: nothing titrates
    AllChem.EmbedMolecule(mol, randomSeed=1)
    ligand = tmp_path / "ligand.sdf"
    Chem.MolToMolFile(mol, str(ligand))

    [only] = _enumerate_protomers(ligand, 7.0, 3.0, _OneChargeState())

    assert (only.charge, only.intrinsic_kcal) == (0, 0.0)
    assert only.mol.GetNumAtoms() == mol.GetNumAtoms()
    assert only.mol.GetNumConformers() == 1
