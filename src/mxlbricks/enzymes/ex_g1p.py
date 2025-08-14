"""name

Equilibrator
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.utils import (
    default_name,
    default_vmax,
    default_km,
    default_kis,
    default_kas,
    filter_stoichiometry,
    static,
)
from mxlpy import units

if TYPE_CHECKING:
    from mxlpy import Model


def _rate_starch(
    g1p: float,
    atp: float,
    adp: float,
    pi: float,
    pga: float,
    f6p: float,
    fbp: float,
    v_st: float,
    kmst1: float,
    kmst2: float,
    ki_st: float,
    kast1: float,
    kast2: float,
    kast3: float,
) -> float:
    return (
        v_st
        * g1p
        * atp
        / (
            (g1p + kmst1)
            * (
                (1 + adp / ki_st) * (atp + kmst2)
                + kmst2 * pi / (kast1 * pga + kast2 * f6p + kast3 * fbp)
            )
        )
    )


def add_g1p_efflux(
    model: Model,
    *,
    rxn: str | None = None,
    g1p: str | None = None,
    atp: str | None = None,
    adp: str | None = None,
    pi: str | None = None,
    pga: str | None = None,
    f6p: str | None = None,
    fbp: str | None = None,
    starch: str | None = None,
    kcat: str | None = None,
    e0: str | None = None,
    km_g1p: str | None = None,
    km_atp: str | None = None,
    ki: str | None = None,
    ka_pga: str | None = None,
    ka_f6p: str | None = None,
    ka_fbp: str | None = None,
) -> Model:
    rxn = default_name(rxn, n.ex_g1p)
    g1p = default_name(g1p, n.g1p)
    atp = default_name(atp, n.atp)
    adp = default_name(adp, n.adp)
    pi = default_name(pi, n.pi)
    pga = default_name(pga, n.pga)
    f6p = default_name(f6p, n.f6p)
    fbp = default_name(fbp, n.fbp)
    starch = default_name(starch, n.starch)

    model.add_reaction(
        name=rxn,
        fn=_rate_starch,
        stoichiometry=filter_stoichiometry(
            model,
            {
                g1p: -1.0,
                atp: -1.0,
                adp: 1.0,
            },
            optional={
                starch: 1.0,
            },
        ),
        args=[
            g1p,
            atp,
            adp,
            pi,
            pga,
            f6p,
            fbp,
            default_vmax(
                model,
                rxn=rxn,
                e0=e0,
                kcat=kcat,
                e0_value=1.0,  # Source
                kcat_value=0.04 * 8,  # Source
            ),
            default_km(model, par=km_g1p, rxn=rxn, subs=g1p, value=0.08, unit=units.mmol / units.liter),
            default_km(model, par=km_atp, rxn=rxn, subs=atp, value=0.08, unit=units.mmol / units.liter),
            default_kis(model, par=ki, rxn=rxn, substrate=adp, value=10, unit=units.mmol / units.liter),
            default_kas(model, par=ka_pga, rxn=rxn, substrate=pga, value=0.1, unit=units.mmol / units.liter),
            default_kas(model, par=ka_f6p, rxn=rxn, substrate=f6p, value=0.02, unit=units.mmol / units.liter),
            default_kas(model, par=ka_fbp, rxn=rxn, substrate=fbp, value=0.02, unit=units.mmol / units.liter),
        ],
    )
    return model
