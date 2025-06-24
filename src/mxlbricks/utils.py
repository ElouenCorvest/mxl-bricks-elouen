from collections.abc import Callable, Mapping

import sympy
from mxlpy import Derived, Model
from mxlpy.fns import mul

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s


def static(
    model: Model,
    name: str,
    value: float,
    unit: sympy.Expr | None = None,
    source: str | None = None,
) -> str:
    model.add_parameter(name, value, unit=unit, source=source)
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


def default_par(
    model: Model,
    *,
    par: str | None,
    name: str,
    value: float,
    unit: sympy.Expr | None = None,
    source: str | None = None,
) -> str:
    if par is not None:
        return par
    return static(model=model, name=name, value=value, unit=unit, source=source)


def default_keq(
    model: Model,
    *,
    par: str | None,
    rxn: str,
    value: float,
    unit: sympy.Expr | None = None,
    source: str | None = None,
) -> str:
    return default_par(
        model=model,
        par=par,
        value=value,
        unit=unit,
        source=source,
        name=n.keq(rxn),
    )


def default_kf(
    model: Model,
    *,
    par: str | None,
    rxn: str,
    value: float,
    unit: sympy.Expr | None = None,
    source: str | None = None,
) -> str:
    return default_par(
        model=model,
        par=par,
        value=value,
        unit=unit,
        source=source,
        name=n.kf(rxn),
    )


def default_km(
    model: Model,
    *,
    par: str | None,
    rxn: str,
    subs: str,
    value: float,
    unit: sympy.Expr | None = None,
    source: str | None = None,
) -> str:
    return default_par(
        model=model,
        par=par,
        value=value,
        unit=unit,
        source=source,
        name=n.km(rxn, subs),
    )


def default_kms(
    model: Model,
    *,
    par: str | None,
    rxn: str,
    value: float,
    unit: sympy.Expr | None = None,
    source: str | None = None,
) -> str:
    return default_par(
        model=model,
        par=par,
        value=value,
        unit=unit,
        source=source,
        name=n.kms(rxn),
    )


def default_kmp(
    model: Model,
    *,
    par: str | None,
    rxn: str,
    value: float,
    unit: sympy.Expr | None = None,
    source: str | None = None,
) -> str:
    return default_par(
        model=model,
        par=par,
        value=value,
        unit=unit,
        source=source,
        name=n.kmp(rxn),
    )


def default_ki(
    model: Model,
    *,
    par: str | None,
    rxn: str,
    value: float,
    unit: sympy.Expr | None = None,
    source: str | None = None,
) -> str:
    return default_par(
        model=model,
        par=par,
        value=value,
        unit=unit,
        source=source,
        name=n.ki(rxn),
    )


def default_kis(
    model: Model,
    *,
    par: str | None,
    rxn: str,
    substrate: str,
    value: float,
    unit: sympy.Expr | None = None,
    source: str | None = None,
) -> str:
    return default_par(
        model=model,
        par=par,
        value=value,
        unit=unit,
        source=source,
        name=n.ki(rxn, substrate),
    )


def default_kre(
    model: Model,
    *,
    par: str | None,
    rxn: str,
    value: float,
    unit: sympy.Expr | None = None,
    source: str | None = None,
) -> str:
    return default_par(
        model=model,
        par=par,
        value=value,
        unit=unit,
        source=source,
        name=n.kre(rxn),
    )


def default_e0(
    model: Model,
    *,
    par: str | None,
    rxn: str,
    value: float,
    unit: sympy.Expr | None = None,
    source: str | None = None,
) -> str:
    return default_par(
        model=model,
        par=par,
        value=value,
        unit=unit,
        source=source,
        name=n.e0(rxn),
    )


def default_kcat(
    model: Model,
    *,
    par: str | None,
    rxn: str,
    value: float,
    unit: sympy.Expr | None = None,
    source: str | None = None,
) -> str:
    return default_par(
        model=model,
        par=par,
        value=value,
        unit=unit,
        source=source,
        name=n.kcat(rxn),
    )


def default_vmax(
    model: Model,
    *,
    e0: str | None,
    kcat: str | None,
    rxn: str,
    e0_value: float,
    e0_unit: sympy.Expr | None = None,
    e0_source: str | None = None,
    kcat_value: float,
    kcat_unit: sympy.Expr | None = None,
    kcat_source: str | None = None,
) -> str:
    e0 = default_e0(
        model=model,
        par=e0,
        rxn=rxn,
        value=e0_value,
        unit=e0_unit,
        source=e0_source,
    )
    kcat = default_kcat(
        model=model,
        par=kcat,
        rxn=rxn,
        value=kcat_value,
        unit=kcat_unit,
        source=kcat_source,
    )
    model.add_derived(vmax := n.vmax(rxn), fn=mass_action_1s, args=[kcat, e0])
    return vmax
