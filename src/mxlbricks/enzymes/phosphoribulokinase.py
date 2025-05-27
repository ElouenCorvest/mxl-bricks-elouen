"""phosphoribulokinase

EC 2.7.1.19

Equilibrator
    ATP(aq) + D-Ribulose 5-phosphate(aq) ⇌ ADP(aq) + D-Ribulose 1,5-bisphosphate(aq)
    Keq = 1e5 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.phosphoribulokinase()


def _rate_prk(
    ru5p: float,
    atp: float,
    pi: float,
    pga: float,
    rubp: float,
    adp: float,
    v13: float,
    km131: float,
    km132: float,
    ki131: float,
    ki132: float,
    ki133: float,
    ki134: float,
    ki135: float,
) -> float:
    return (
        v13
        * ru5p
        * atp
        / (
            (ru5p + km131 * (1 + pga / ki131 + rubp / ki132 + pi / ki133))
            * (atp * (1 + adp / ki134) + km132 * (1 + adp / ki135))
        )
    )


def add_phosphoribulokinase(
    model: Model,
    *,
    chl_stroma: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    km_ru5p: str | None = None,
    km_atp: str | None = None,
    ki1: str | None = None,
    ki2: str | None = None,
    ki3: str | None = None,
    ki4: str | None = None,
    ki5: str | None = None,
) -> Model:
    km_ru5p = (
        static(model, n.km(ENZYME, n.ru5p()), 0.05) if km_ru5p is None else km_ru5p
    )  # FIXME: source
    km_atp = (
        static(model, n.km(ENZYME, n.atp()), 0.05) if km_atp is None else km_atp
    )  # FIXME: source
    ki1 = static(model, n.ki(ENZYME, n.pga()), 2.0) if ki1 is None else ki1
    ki2 = static(model, n.ki(ENZYME, n.rubp()), 0.7) if ki2 is None else ki2
    ki3 = static(model, n.ki(ENZYME, n.pi()), 4.0) if ki3 is None else ki3
    ki4 = static(model, n.ki(ENZYME, "4"), 2.5) if ki4 is None else ki4
    ki5 = static(model, n.ki(ENZYME, "5"), 0.4) if ki5 is None else ki5
    kcat = (
        static(model, n.kcat(ENZYME), 0.9999 * 8) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=_rate_prk,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.ru5p(chl_stroma): -1.0,
                n.atp(chl_stroma): -1.0,
                n.rubp(chl_stroma): 1.0,
                n.adp(chl_stroma): 1.0,
            },
        ),
        args=[
            n.ru5p(chl_stroma),
            n.atp(chl_stroma),
            n.pi(chl_stroma),
            n.pga(chl_stroma),
            n.rubp(chl_stroma),
            n.adp(chl_stroma),
            vmax,
            km_ru5p,
            km_atp,
            ki1,
            ki2,
            ki3,
            ki4,
            ki5,
        ],
    )
    return model
