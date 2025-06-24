from collections.abc import Callable, Mapping

from mxlpy import Derived, Model
from mxlpy.fns import mul

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s


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


def default_name(name: str | None, name_fn: Callable[[], str]) -> str:
    if name is None:
        return name_fn()
    return name


def default_par(model: Model, *, par: str | None, name: str, value: float) -> str:
    if par is not None:
        return par
    return static(model=model, name=name, value=value)


def default_keq(model: Model, *, par: str | None, rxn: str, default: float) -> str:
    if par is not None:
        return par
    return static(model=model, name=n.keq(rxn), value=default)


def default_kf(model: Model, *, par: str | None, rxn: str, default: float) -> str:
    if par is not None:
        return par
    return static(model=model, name=n.kf(rxn), value=default)


def default_km(
    model: Model, *, par: str | None, rxn: str, subs: str, default: float
) -> str:
    if par is not None:
        return par
    return static(model=model, name=n.km(rxn, subs), value=default)


def default_kms(model: Model, *, par: str | None, rxn: str, default: float) -> str:
    if par is not None:
        return par
    return static(model=model, name=n.kms(rxn), value=default)


def default_kmp(model: Model, *, par: str | None, rxn: str, default: float) -> str:
    if par is not None:
        return par
    return static(model=model, name=n.kmp(rxn), value=default)


def default_ki(model: Model, *, par: str | None, rxn: str, default: float) -> str:
    if par is not None:
        return par
    return static(model=model, name=n.ki(rxn), value=default)


def default_kis(
    model: Model, *, par: str | None, rxn: str, substrate: str, default: float
) -> str:
    if par is not None:
        return par
    return static(model=model, name=n.ki(rxn, substrate), value=default)


def default_kre(model: Model, *, par: str | None, rxn: str, default: float) -> str:
    if par is not None:
        return par
    return static(model=model, name=n.kre(rxn), value=default)


def default_e0(model: Model, *, par: str | None, rxn: str, default: float) -> str:
    if par is not None:
        return par
    return static(model=model, name=n.e0(rxn), value=default)


def default_kcat(model: Model, *, par: str | None, rxn: str, default: float) -> str:
    if par is not None:
        return par
    return static(model=model, name=n.kcat(rxn), value=default)


def default_vmax(
    model: Model,
    e0: str | None,
    kcat: str | None,
    rxn: str,
    e0_default: float,
    kcat_default: float,
) -> str:
    e0 = default_e0(model=model, par=e0, rxn=rxn, default=e0_default)
    kcat = default_kcat(model=model, par=kcat, rxn=rxn, default=kcat_default)
    model.add_derived(vmax := n.vmax(rxn), fn=mass_action_1s, args=[kcat, e0])
    return vmax
