"""Aspartate aminotransferase

EC 2.6.1.1

Equilibrator
Aspartate(aq) + alpha-Ketoglutarate(aq) ⇌ Oxaloacetate(aq) + Glutamate(aq)
Keq = 0.3 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, reversible_michaelis_menten_2s_2p
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.aspartate_aminotransferase()


def add_aspartate_aminotransferase(
    model: Model,
    compartment: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 2.53) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 3.88) if kmp is None else kmp  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 84) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 0.3) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    stoichiometry = filter_stoichiometry(
        model,
        {
            n.aspartate(compartment): -1.0,
            n.oxoglutarate(compartment): -1.0,
            n.oxaloacetate(compartment): 1.0,
            n.glutamate(compartment): 1.0,
        },
    )

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_2p,
        stoichiometry=stoichiometry,
        args=[
            n.aspartate(compartment),
            n.oxoglutarate(compartment),
            n.oxaloacetate(compartment),
            n.glutamate(compartment),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
