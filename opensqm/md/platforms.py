"""Process-wide OpenMM compute-platform selection.

OpenMM auto-picks the fastest available platform whenever a ``Context`` or
``Simulation`` is built. A CLI run can override that here: :func:`set_platform`
records a forced platform once at start-up, and every simulation built through
:func:`make_simulation` / :func:`make_context` is pinned to it. This lets
``run_mmgbsa`` / ``run_modbind`` guarantee they run on the GPU (CUDA on a Linux
GPU node, Metal/OpenCL on Apple Silicon) rather than silently dropping to the
CPU platform when the fast one fails to initialise.

When no platform is forced (the default), the helpers build contexts exactly as
before and leave OpenMM to choose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import openmm
from loguru import logger
from openmm.app import Simulation

if TYPE_CHECKING:
    from openmm.app.topology import Topology

# User-facing platform names -> ordered OpenMM platform names to try. The first
# one OpenMM actually has registered wins, so ``"mps"`` works whether or not the
# optional ``openmm-metal`` plugin (which registers the "Metal" platform) is
# installed, falling back to Apple's OpenCL backend otherwise.
_PLATFORM_ALIASES: dict[str, tuple[str, ...]] = {
    "cuda": ("CUDA",),
    "mps": ("Metal", "OpenCL"),
    "metal": ("Metal", "OpenCL"),
    "opencl": ("OpenCL",),
    "cpu": ("CPU",),
    "reference": ("Reference",),
}

# Mutable holder for the forced platform (a plain module global would need the
# ``global`` statement to rebind); ``None`` means "let OpenMM auto-select".
_state: dict[str, openmm.Platform | None] = {"forced": None}


def _available_platforms() -> set[str]:
    return {
        openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())
    }


def _resolve(name: str) -> openmm.Platform:
    candidates = _PLATFORM_ALIASES.get(name.strip().lower(), (name,))
    available = _available_platforms()
    for candidate in candidates:
        if candidate in available:
            return openmm.Platform.getPlatformByName(candidate)
    raise ValueError(
        f"No OpenMM platform available for '{name}' (tried {list(candidates)}). "
        f"Available platforms: {sorted(available)}."
    )


def set_platform(name: str | None) -> openmm.Platform | None:
    """Force every subsequently built context onto platform ``name`` (or clear it).

    ``name`` is a user-facing alias (``"cuda"``, ``"mps"``) or an exact OpenMM
    platform name; ``None`` restores OpenMM's automatic selection. Raises
    ``ValueError`` when no matching platform is registered, so a forced run
    fails loudly instead of silently falling back to the CPU.
    """
    if name is None:
        _state["forced"] = None
        return None
    platform = _resolve(name)
    _state["forced"] = platform
    logger.info(f"Forcing OpenMM platform: {platform.getName()}")
    return platform


def get_platform() -> openmm.Platform | None:
    """Return the forced platform, or ``None`` when OpenMM should auto-select."""
    return _state["forced"]


def make_context(system: openmm.System, integrator: openmm.Integrator) -> openmm.Context:
    """Build a ``Context``, pinned to the forced platform when one is set."""
    platform = _state["forced"]
    if platform is not None:
        return openmm.Context(system, integrator, platform)
    return openmm.Context(system, integrator)


def make_simulation(
    topology: Topology, system: openmm.System, integrator: openmm.Integrator
) -> Simulation:
    """Build a ``Simulation``, pinned to the forced platform when one is set."""
    platform = _state["forced"]
    if platform is not None:
        return Simulation(topology, system, integrator, platform)
    return Simulation(topology, system, integrator)
