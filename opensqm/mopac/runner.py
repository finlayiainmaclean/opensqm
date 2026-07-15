"""Invoke the MOPAC executable and validate that a job completed successfully."""

import os
import subprocess
from pathlib import Path

from opensqm.mopac.exceptions import MOPACError

MOPAC_BIN = os.getenv("MOPAC_BIN", "mopac")


def _run_mopac_input_file(mopac_input: Path, *, cwd: Path) -> None:
    """Run MOPAC on a control file via a list subprocess (no shell).

    Mirrors ``pymopac.MopacInput.silentRun``.
    """
    result = subprocess.run(
        [MOPAC_BIN, str(mopac_input)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise MOPACError(
            f"MOPAC failed (exit {result.returncode}, exe={MOPAC_BIN!r})\n"
            f"stdout:\n{result.stdout}\nostderr:\n{result.stderr}"
        )


def check_mopac_was_success(output_str: str) -> None:
    """Raise ``MOPACError`` unless the output shows a normal, successful MOPAC run."""
    if "IMAGINARY FREQUENCIES" in output_str:
        raise MOPACError(f"IMAGINARY FREQUENCIES: {output_str}")
    if "EXCESS NUMBER OF OPTIMIZATION CYCLES" in output_str:
        raise MOPACError(f"EXCESS NUMBER OF OPTIMIZATION CYCLES: {output_str}")
    if "NOT ENOUGH TIME FOR ANOTHER CYCLE" in output_str:
        raise MOPACError(f"NOT ENOUGH TIME FOR ANOTHER CYCLE: {output_str}")
    success_keys = ["JOB ENDED NORMALLY", "MOPAC DONE"]
    correct_keys = all(key in output_str for key in success_keys)
    if correct_keys:
        return
    elif "A hydrogen atom is badly positioned" in output_str:
        raise MOPACError(f"Bad hydrogen: {output_str}")
    else:
        raise MOPACError(f"Unknown error: {output_str}")
