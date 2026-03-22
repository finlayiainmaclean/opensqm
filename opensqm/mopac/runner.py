# ruff: noqa: D100, D103, E501
import subprocess
from pathlib import Path

from opensqm.mopac.exceptions import MOPACError


def _mopac_executable() -> str:
    """Resolve MOPAC CLI the same way pymopac's MopacInput does (see pymopac.input.silentRun)."""
    from pymopac.helpers import get_mopac  # noqa: PLC0415

    return get_mopac() or "mopac"


def _run_mopac_input_file(mopac_input: Path, *, cwd: Path) -> None:
    """Run MOPAC on a control file using list subprocess (no shell), matching pymopac.MopacInput.silentRun."""
    exe = _mopac_executable()
    result = subprocess.run(
        [exe, str(mopac_input)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise MOPACError(
            f"MOPAC failed (exit {result.returncode}, exe={exe!r})\n"
            f"stdout:\n{result.stdout}\nostderr:\n{result.stderr}"
        )


def check_mopac_was_success(output_str: str) -> None:
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
