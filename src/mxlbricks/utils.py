from collections.abc import Mapping

from mxlpy import Derived, Model
from mxlpy.fns import mul

from mxlbricks import names as n


def static(
    model: Model,
    name: str,
    value: float,
    unit: str | None = None,  # noqa: ARG001
) -> str:
    model.add_parameter(name, value)
    return name


def fcbb_regulated(model: Model, name: str, value: float) -> str:
    new_name = f"{name}_fcbb"

    model.add_parameter(name, value)
    model.add_derived(new_name, mul, args=[name, n.light_speedup()])
    return new_name


def thioredixon_regulated(model: Model, name: str, value: float) -> str:
    new_name = f"{name}_active"
    model.add_parameter(name, value)
    model.add_derived(new_name, mul, args=[name, n.e_active()])
    return new_name


def filter_stoichiometry(
    model: Model,
    stoichiometry: Mapping[str, float | Derived],
    optional: dict[str, float] | None = None,
) -> Mapping[str, float | Derived]:
    """Only use components that are actually compounds in the model"""
    variables = model.get_raw_variables(as_copy=False)

    new = {}
    for k, v in stoichiometry.items():
        if k in variables:
            new[k] = v
        elif k not in model._ids:  # noqa: SLF001
            msg = f"Missing component {k}"
            raise KeyError(msg)

    optional = {} if optional is None else optional
    new |= {k: v for k, v in optional.items() if k in variables}
    return new
