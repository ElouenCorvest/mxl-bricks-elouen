"""A5P <=> RU5P

EC 5.3.1.13

Equilibrator
Arabinose-5-phosphate <=> Ru5P
Keq = 0.4 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, reversible_michaelis_menten_1s_1p
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.a5p_isomerase()


def add_a5p_isomerase(
    model: Model,
    compartment: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = (
        static(model, n.kms(ENZYME), 1.89) if kms is None else kms
    )  # Clostridium tetani
    kmp = (
        static(model, n.kmp(ENZYME), 6.65) if kmp is None else kmp
    )  # Clostridium tetani
    kcat = (
        static(model, n.kcat(ENZYME), 102) if kcat is None else kcat
    )  # Clostridium tetani
    e0 = static(model, n.e0(ENZYME), 1) if e0 is None else e0  # Clostridium tetani
    keq = static(model, n.keq(ENZYME), 0.4) if keq is None else keq
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    stoichiometry = filter_stoichiometry(
        model,
        {
            n.arabinose_5_phosphate(compartment): -1.0,
            n.ru5p(compartment): 1.0,
        },
    )

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_1s_1p,
        stoichiometry=stoichiometry,
        args=[
            n.arabinose_5_phosphate(compartment),
            n.ru5p(compartment),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
