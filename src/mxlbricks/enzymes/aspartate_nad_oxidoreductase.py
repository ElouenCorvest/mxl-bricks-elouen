"""Aspartate:NAD oxidoreductase

EC FIXME

Equilibrator
Iminoaspartate(aq) + NADPH(aq) ⇌ Aspartate(aq) + NADP(aq)
Keq = 1.6e10 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, michaelis_menten_2s
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.aspartate_oxidoreductase()


def add_aspartate_oxidoreductase(
    model: Model,
    compartment: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.1) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1.0) if kmp is None else kmp  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 1.0) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    stoichiometry = filter_stoichiometry(
        model,
        {
            n.iminoaspartate(compartment): -1.0,
            n.nadh(compartment): -1.0,
            n.aspartate(compartment): 1.0,
            n.nad(compartment): 1.0,
        },
    )

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_2s,
        stoichiometry=stoichiometry,
        args=[
            n.iminoaspartate(compartment),
            n.nadp(compartment),
            vmax,
            kms,
        ],
    )
    return model
