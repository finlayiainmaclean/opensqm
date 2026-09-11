"""CTMD: hit triage from the reversible work needed to unseat a docked pose.

Implements the method of Shekhar et al. (bioRxiv 2026,
doi:10.64898/2026.02.05.703972): well-tempered metadynamics biases the ligand
RMSD from its docked pose until the ligand leaves, and the Tiwary-Parrinello
bias offset c(t) at the moment it leaves ranks the ligand. Only the bound state
is simulated, so a ligand costs a few nanoseconds rather than a full binding
free energy.

The driver lives in ``opensqm.ctmd.run_ctmd`` and is deliberately not re-exported
here: it pulls in the whole system-preparation stack, which costs about ninety
seconds of import time that scoring an existing run does not need.
"""

from opensqm.ctmd.config import CTMDSettings
from opensqm.ctmd.metad import (
    CTMDTrajectory,
    build_metadynamics,
    ct,
    ct_from_bias,
    run_ctmd_replica,
)
from opensqm.ctmd.score import bootstrap_score, commit_frame, rank_ligands, score_replica

__all__ = [
    "CTMDSettings",
    "CTMDTrajectory",
    "bootstrap_score",
    "build_metadynamics",
    "commit_frame",
    "ct",
    "ct_from_bias",
    "rank_ligands",
    "run_ctmd_replica",
    "score_replica",
]
