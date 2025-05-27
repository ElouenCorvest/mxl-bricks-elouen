"""name

Equilibrator
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.ex_g1p()


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
    chl_stroma: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    km_g1p: str | None = None,
    km_atp: str | None = None,
    ki: str | None = None,
    ka_pga: str | None = None,
    ka_f6p: str | None = None,
    ka_fbp: str | None = None,
) -> Model:
    kcat = (
        static(model, n.kcat(ENZYME), 0.04 * 8) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    km_g1p = static(model, n.km(ENZYME, n.g1p()), 0.08) if km_g1p is None else km_g1p
    km_atp = static(model, n.km(ENZYME, n.atp()), 0.08) if km_atp is None else km_atp
    ki = static(model, n.ki(ENZYME), 10.0) if ki is None else ki
    ka_pga = static(model, n.ka(ENZYME, n.pga()), 0.1) if ka_pga is None else ka_pga
    ka_f6p = static(model, n.ka(ENZYME, n.f6p()), 0.02) if ka_f6p is None else ka_f6p
    ka_fbp = static(model, n.ka(ENZYME, n.fbp()), 0.02) if ka_fbp is None else ka_fbp

    stoichiometry = filter_stoichiometry(
        model,
        {
            n.g1p(chl_stroma): -1.0,
            n.atp(chl_stroma): -1.0,
            n.adp(chl_stroma): 1.0,
        },
        optional={
            n.starch(chl_stroma): 1.0,
        },
    )

    model.add_reaction(
        name=ENZYME,
        fn=_rate_starch,
        stoichiometry=stoichiometry,
        args=[
            n.g1p(chl_stroma),
            n.atp(chl_stroma),
            n.adp(chl_stroma),
            n.pi(chl_stroma),
            n.pga(chl_stroma),
            n.f6p(chl_stroma),
            n.fbp(chl_stroma),
            vmax,
            km_g1p,
            km_atp,
            ki,
            ka_pga,
            ka_f6p,
            ka_fbp,
        ],
    )
    return model
