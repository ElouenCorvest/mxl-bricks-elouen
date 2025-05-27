"""glycerate dehydrogenase

NADH + Hydroxypyruvate <=> NAD  + D-Glycerate

Equilibrator
NADH(aq) + Hydroxypyruvate(aq) ⇌ NAD(aq) + D-Glycerate(aq)
Keq = 8.7e4 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    michaelis_menten_1s,
    reversible_michaelis_menten_2s_2p,
)
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.glycerate_dehydrogenase()


def add_hpa_outflux(
    model: Model,
    *,
    compartment: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.12) if kms is None else kms  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 398.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_1s,
        stoichiometry={
            n.hydroxypyruvate(compartment): -1.0,
        },
        args=[
            n.hydroxypyruvate(compartment),
            vmax,
            kms,
        ],
    )

    return model


def add_glycerate_dehydrogenase(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.12) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1.0) if kmp is None else kmp  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 398.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 87000.0) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_2p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.nadh(): -1.0,
                n.hydroxypyruvate(): -1.0,
                n.nad(): 1.0,
                n.glycerate(): 1.0,
            },
        ),
        args=[
            n.hydroxypyruvate(),
            n.nadh(),
            n.glycerate(),
            n.nad(),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )

    return model
