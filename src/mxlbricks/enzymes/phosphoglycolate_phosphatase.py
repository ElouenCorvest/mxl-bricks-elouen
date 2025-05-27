"""phosphoglycolate phosphatase, EC 3.1.3.18

H2O(chl) + PGO(chl) <=> Orthophosphate(chl) + Glycolate(chl)

Equilibrator
H2O(l) + PGO(aq) ⇌ Orthophosphate(aq) + Glycolate(aq)
Keq = 3.1e5 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    reversible_michaelis_menten_1s_1p_1i,
    value,
)
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.phosphoglycolate_phosphatase()


def add_phosphoglycolate_influx(
    model: Model,
    *,
    chl_stroma: str = "",
    kf: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 60.0) if kf is None else kf

    model.add_reaction(
        name=ENZYME,
        fn=value,
        stoichiometry={
            n.glycolate(chl_stroma): 1,
        },
        args=[
            kf,
        ],
    )
    return model


def add_phosphoglycolate_phosphatase(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
    ki_pi: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.029) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1.0) if kmp is None else kmp  # FIXME: source
    ki_pi = static(model, n.ki(ENZYME, n.pi()), 12.0)
    kcat = (
        static(model, n.kcat(ENZYME), 292.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = (
        static(model, n.keq(ENZYME), 310000.0) if keq is None else keq
    )  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_1s_1p_1i,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.pgo(): -1.0,
                n.glycolate(): 1.0,
                n.pi(): 1.0,
            },
        ),
        args=[
            n.pgo(),
            n.glycolate(),
            n.pi(),
            vmax,
            kms,
            kmp,
            ki_pi,
            keq,
        ],
    )
    return model
