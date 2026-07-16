"""Top-level ``opensqm`` command-line interface.

Exposed as the ``opensqm`` console script (see ``[project.scripts]`` in
``pyproject.toml``). Each subcommand is a thin wrapper around a module-level
runner's click command; register a new one by importing its command and adding
it to :data:`cli` below.

    opensqm mmgbsa --protein prot.pdb --ligand lig.sdf --output run/
"""

from __future__ import annotations

import click

from opensqm.md.run_mmgbsa import main as mmgbsa_command
from opensqm.modbind.run_modbind import main as modbinddg_command


@click.group()
def cli() -> None:
    """Command-line tools for opensqm."""


cli.add_command(modbinddg_command, name="modbind")
cli.add_command(mmgbsa_command, name="mmgbsa")


if __name__ == "__main__":
    cli()
