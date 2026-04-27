"""Utility functions."""

import subprocess
from pathlib import Path

from loguru import logger

DATA_DIR = Path(__file__).parent.parent / "data"
LIGAND_FORCEFIELD_DIR = DATA_DIR / "ligand_forcefields"
LIGAND_FORCEFIELD_DIR.mkdir(parents=True, exist_ok=True)


def run_command(
    command: str,
    shell: bool = True,
    ignore_errors: bool = False,
    cwd: str | Path | None = None,
):
    """Thin wrapper around subprocess.run that logs output and optionally raises."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            shell=shell,
            cwd=str(cwd) if cwd is not None else None,
        )
        logger.debug(result.stdout)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if not ignore_errors:
            raise Exception(
                f"Command failed: {command}\nstdout: {e.stdout.strip()}\nstderr: {e.stderr.strip()}"
            ) from e


if __name__ == "__main__":
    print(DATA_DIR)
