"""Empirical pKa prediction with PROPKA.

PROPKA assigns an empirical pKa to every ionisable group in a protein from a
single static structure. Unlike the MD-derived fits in :mod:`opensqm.cph.pka`,
these predictions are cheap and effectively pH-independent: the ``pH`` argument
only sets the reference point reported in PROPKA's protein-stability output, it
does not change the predicted pKa values.

The main entry point, :func:`find_titratable_residues_near_ph`, is a convenient
way to decide which residues are worth driving with constant-pH MC. Rather than
a fixed pKa window, it scores each group by the pH-adjusted free energy
(kcal/mol) of its minor protonation state and keeps the ones below a penalty -
the residue analogue of the ligand protomer filter in :mod:`opensqm.run_cph`,
which keeps protomers with
``relative_ph_adjusted_free_energy < ligand_protonation_penalty`` (the ligand
filter works in reduced kT units; here the energy is in true kcal/mol).
"""

from __future__ import annotations

import io
import math
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from openmm import unit

# Third-party PROPKA distribution. This module is ``opensqm.cph.propka``; the
# absolute import below resolves to the installed ``propka`` package, not to
# this file.
from propka.run import single

# ln(10): the deprotonation free energy of a group at a given pH is
# R T ln(10) (pKa - pH), so ln(10) is the per-pH-unit slope of the pKa-to-free
# energy conversion (before multiplying by R T).
_LN10 = math.log(10.0)


@dataclass(frozen=True)
class PredictedPka:
    """A single PROPKA pKa prediction for one ionisable group."""

    residue_name: str
    residue_number: int
    chain_id: str
    pka: float

    @property
    def label(self) -> str:
        """Human-readable ``"<resname> <resnum> <chain>"`` identifier."""
        chain = f" {self.chain_id}" if self.chain_id else ""
        return f"{self.residue_name} {self.residue_number}{chain}"

    def relative_ph_adjusted_free_energy(
        self, pH: float, *, temperature: unit.Quantity = 298.15 * unit.kelvin
    ) -> float:
        """pH-adjusted free energy of this group's minor protonation state.

        Returns the free energy (kcal/mol) of the less-populated protonation
        state relative to the dominant one at ``pH``. It is zero when
        ``pH == pka`` (both states equally populated) and grows by
        ``R T ln(10)`` (about 1.36 kcal/mol at 298.15 K) per pH unit of
        separation, i.e. ``R T ln(10) * abs(pka - pH)``.

        Parameters
        ----------
        pH : float
            pH at which to evaluate the protonation penalty.
        temperature : openmm.unit.Quantity, optional
            Temperature for the ``R T`` prefactor. Defaults to 298.15 K.

        Returns
        -------
        float
            The (non-negative) relative free energy in kcal/mol.
        """
        rt_ln10 = (unit.MOLAR_GAS_CONSTANT_R * temperature * _LN10).value_in_unit(
            unit.kilocalories_per_mole
        )
        return rt_ln10 * abs(self.pka - pH)


def predict_pkas(pdb_path: str | Path, *, pH: float = 7.0) -> list[PredictedPka]:
    """Run PROPKA on a protein PDB and return every predicted group pKa.

    Parameters
    ----------
    pdb_path : str | Path
        Path to the protein structure. PROPKA expects a PDB file with
        hydrogens stripped or present; either works.
    pH : float, optional
        pH passed to PROPKA (its ``--pH`` option). This only affects the
        ancillary protein-stability/charge output, not the predicted pKa
        values themselves.

    Returns
    -------
    list[PredictedPka]
        One entry per ionisable group PROPKA found (titratable side chains,
        the N- and C-termini, and any ligand groups), in the order PROPKA
        reports them.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)

    # PROPKA writes its full report to stdout; capture it so it does not leak
    # into the caller's logs, and disable the side-effect ``.pka`` file.
    with redirect_stdout(io.StringIO()):
        molecular_container = single(str(pdb_path), optargs=["--pH", str(pH)], write_pka=False)

    conformations = molecular_container.conformations
    # PROPKA stores an averaged conformation under "AVR"; fall back to whatever
    # single conformation exists for structures with no alternate locations.
    conformation = conformations.get("AVR") or next(iter(conformations.values()))

    predictions: list[PredictedPka] = []
    for group in conformation.groups:
        if group.pka_value is None:
            continue
        atom = group.atom
        predictions.append(
            PredictedPka(
                residue_name=atom.res_name.strip(),
                residue_number=int(atom.res_num),
                chain_id=(atom.chain_id or "").strip(),
                pka=float(group.pka_value),
            )
        )
    return predictions


def find_titratable_residues_near_ph(
    pdb_path: str | Path,
    *,
    pH: float = 7.0,
    protonation_penalty: unit.Quantity = 3.0 * unit.kilocalories_per_mole,
    temperature: unit.Quantity = 298.15 * unit.kelvin,
) -> list[PredictedPka]:
    """Residues whose minor protonation state is accessible at ``pH``.

    Runs PROPKA on ``pdb_path`` and keeps every ionisable group whose minor
    protonation state carries a pH-adjusted free energy below
    ``protonation_penalty``, where that free energy is
    ``R T ln(10) * abs(pKa - pH)`` in kcal/mol (see
    :meth:`PredictedPka.relative_ph_adjusted_free_energy`).

    This mirrors the ligand protomer filter in :mod:`opensqm.run_cph`
    (``relative_ph_adjusted_free_energy < ligand_protonation_penalty``), but is
    evaluated in true kcal/mol rather than the reduced (kT) units the ligand
    filter uses, so the same nominal penalty maps to a slightly wider pKa
    window here.

    Parameters
    ----------
    pdb_path : str | Path
        Path to the protein structure.
    pH : float, optional
        Target pH. Defaults to 7.0.
    protonation_penalty : openmm.unit.Quantity, optional
        Maximum pH-adjusted free energy of the minor protonation state for a
        group to be kept. Defaults to 3.0 kcal/mol, which at 298.15 K
        corresponds to a pKa within about +/-2.2 of ``pH``.
    temperature : openmm.unit.Quantity, optional
        Temperature for the ``R T`` prefactor. Defaults to 298.15 K.

    Returns
    -------
    list[PredictedPka]
        Matching groups, sorted by ascending protonation penalty (the most
        strongly titrating groups first).
    """
    penalty_kcal = protonation_penalty.value_in_unit(unit.kilocalories_per_mole)
    predictions = predict_pkas(pdb_path, pH=pH)

    def penalty(prediction: PredictedPka) -> float:
        return prediction.relative_ph_adjusted_free_energy(pH, temperature=temperature)

    near = [p for p in predictions if penalty(p) < penalty_kcal]
    near.sort(key=penalty)

    logger.info(
        f"PROPKA: {len(near)}/{len(predictions)} groups have a pH-adjusted "
        f"protonation free energy below {penalty_kcal:.2f} kcal/mol at pH {pH}"
    )
    for prediction in near:
        logger.info(
            f"  {prediction.label}: pKa = {prediction.pka:.2f}, "
            f"dG = {penalty(prediction):.2f} kcal/mol"
        )
    return near
