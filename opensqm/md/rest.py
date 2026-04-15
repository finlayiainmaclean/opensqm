"""Prepare a system for REST sampling."""

import collections
import copy
import typing

import numpy as np
from openmm import (
    CustomNonbondedForce,
    CustomTorsionForce,
    HarmonicAngleForce,
    HarmonicBondForce,
    MonteCarloBarostat,
    NonbondedForce,
    PeriodicTorsionForce,
    System,
    unit,
)

_T = typing.TypeVar("_T", bound=NonbondedForce | CustomNonbondedForce)

_SUPPORTED_FORCES = [
    HarmonicBondForce,
    HarmonicAngleForce,
    PeriodicTorsionForce,
    NonbondedForce,
    CustomNonbondedForce,
    MonteCarloBarostat,
]

REST_CTX_PARAM = "bm_b0"
REST_CTX_PARAM_SQRT = "sqrt<bm_b0>"


def _create_torsion_force(force: PeriodicTorsionForce, solute_idxs: set[int]) -> CustomTorsionForce:
    energy_fn = (
        "scale * (k * (1 + cos(periodicity * theta - phase)));"
        f"scale = is_ss * {REST_CTX_PARAM} + is_sw * {REST_CTX_PARAM_SQRT} + is_ww;"
    )

    custom_force = CustomTorsionForce(energy_fn.replace(" ", ""))
    custom_force.addGlobalParameter(REST_CTX_PARAM, 1.0)
    custom_force.addGlobalParameter(REST_CTX_PARAM_SQRT, 1.0)

    for parameter in ["is_ss", "is_sw", "is_ww"]:
        custom_force.addPerTorsionParameter(parameter)

    custom_force.addPerTorsionParameter("periodicity")
    custom_force.addPerTorsionParameter("phase")
    custom_force.addPerTorsionParameter("k")

    custom_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
    custom_force.setForceGroup(force.getForceGroup())

    for i in range(force.getNumTorsions()):
        *idxs, periodicity, phase, k = force.getTorsionParameters(i)

        is_ss = 1.0 if all(idx in solute_idxs for idx in idxs) else 0.0
        is_sw = 1.0 if not is_ss and any(idx in solute_idxs for idx in idxs) else 0.0
        is_ww = 1.0 if not is_ss and not is_sw else 0.0

        custom_force.addTorsion(*idxs, [is_ss, is_sw, is_ww, periodicity, phase, k])

    return custom_force


def _create_default_nonbonded_force(force: NonbondedForce, solute_idxs: set[int]) -> NonbondedForce:
    force.addGlobalParameter(REST_CTX_PARAM, 1.0)
    force.addGlobalParameter(REST_CTX_PARAM_SQRT, 1.0)

    for idx in solute_idxs:
        charge, sigma, epsilon = force.getParticleParameters(idx)

        # We cannot modify magnitude directly using units on 0.0, we just multiply by 0
        if not np.isclose(charge.value_in_unit_system(unit.md_unit_system), 0.0):
            force.addParticleParameterOffset(REST_CTX_PARAM_SQRT, idx, charge, 0.0, 0.0)
            charge *= 0.0

        if not np.isclose(epsilon.value_in_unit_system(unit.md_unit_system), 0.0):
            force.addParticleParameterOffset(REST_CTX_PARAM, idx, 0.0, 0.0, epsilon)
            epsilon *= 0.0

        force.setParticleParameters(idx, charge, sigma, epsilon)

    if force.getNumExceptionParameterOffsets() != 0:
        raise NotImplementedError("exception parameter offsets are not supported")

    for i in range(force.getNumExceptions()):
        *idxs, charge, sigma, epsilon = force.getExceptionParameters(i)

        if all(idx in solute_idxs for idx in idxs):
            ctx_parameter = REST_CTX_PARAM
        elif any(idx in solute_idxs for idx in idxs):
            ctx_parameter = REST_CTX_PARAM_SQRT
        else:
            continue

        if np.isclose(charge.value_in_unit_system(unit.md_unit_system), 0.0) and np.isclose(
            epsilon.value_in_unit_system(unit.md_unit_system), 0.0
        ):
            continue

        force.addExceptionParameterOffset(ctx_parameter, i, charge, sigma * 0.0, epsilon)
        force.setExceptionParameters(i, *idxs, charge * 0.0, sigma, epsilon * 0.0)

    return force


def _create_custom_nonbonded_force(
    force: CustomNonbondedForce,
    solute_idxs: set[int],
) -> CustomNonbondedForce:
    contains_solute_idxs = any(
        len(solute_idxs.union(idxs)) > 0
        for i in range(force.getNumInteractionGroups())
        for idxs in force.getInteractionGroupParameters(i)
    )

    if not contains_solute_idxs:
        return force

    is_solute_param = "is_solute"
    found_parameter_names = {
        force.getPerParticleParameterName(i) for i in range(force.getNumPerParticleParameters())
    }
    assert is_solute_param not in found_parameter_names, "param already present"

    force.addGlobalParameter(REST_CTX_PARAM_SQRT, 1.0)
    force.addPerParticleParameter(is_solute_param)

    energy_fn = force.getEnergyFunction()
    energy_fn = (
        f"scale1 * scale2 * {energy_fn};"
        f"scale1 = 1.0-{is_solute_param}1 + {REST_CTX_PARAM_SQRT} * {is_solute_param}1;"
        f"scale2 = 1.0-{is_solute_param}2 + {REST_CTX_PARAM_SQRT} * {is_solute_param}2;"
    )
    force.setEnergyFunction(energy_fn.replace(" ", "").replace(";;", ";"))

    for idx in range(force.getNumParticles()):
        params = force.getParticleParameters(idx)
        params = [*params, 1.0 if idx in solute_idxs else 0.0]
        force.setParticleParameters(idx, params)

    return force


def _create_nonbonded_force(force: _T, solute_idxs: set[int]) -> _T:
    force = copy.deepcopy(force)
    if isinstance(force, NonbondedForce):
        return _create_default_nonbonded_force(force, solute_idxs)
    elif isinstance(force, CustomNonbondedForce):
        return _create_custom_nonbonded_force(force, solute_idxs)
    raise NotImplementedError


def apply_rest(system: System, solute_idxs: set[int]):
    """Apply REST forces to the system."""
    forces_by_type = collections.defaultdict(dict)

    # Some forces like CMMotionRemover might not need to be scaled
    for i, force in enumerate(system.getForces()):
        if type(force) not in _SUPPORTED_FORCES:
            pass  # ignore unsupported instead of crashing
        forces_by_type[type(force)][i] = force

    rest_forces = {}

    for force_type, forces in forces_by_type.items():
        for i, force in forces.items():
            if force_type == PeriodicTorsionForce:
                new_force = _create_torsion_force(force, solute_idxs)
            elif force_type in (NonbondedForce, CustomNonbondedForce):
                new_force = _create_nonbonded_force(force, solute_idxs)
            else:
                new_force = copy.deepcopy(force)

            rest_forces[i] = new_force

    for i in reversed(range(system.getNumForces())):
        system.removeForce(i)
    for i in range(len(rest_forces)):
        system.addForce(rest_forces[i])
