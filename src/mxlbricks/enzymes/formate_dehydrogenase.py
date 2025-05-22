"""name

EC 1.17.1.9
    Metacyc: 1.2.1.2-RXN
    FORMATE + NAD ⇌ CARBON-DIOXIDE + NADH

    Equilibrator
    NAD (aq) + Formate(aq) + H2O(l) ⇌ NADH(aq) + CO2(total)
    Keq = 8.7e3 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, reversible_michaelis_menten_2s_2p
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.formate_dehydrogenase()


def add_formate_dehydrogenase(
    model: Model,
    per: str,
    mit: str,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.011) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 0.18) if kmp is None else kmp  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 2.9) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 8700.0) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_2p,
        stoichiometry=filter_stoichiometry(
            model,
            stoichiometry={
                n.formate(mit): -1.0,
                n.nad(mit): -1.0,
                n.nadh(mit): 1.0,
                n.co2(mit): 1.0,
            },
        ),
        args=[
            n.nad(per),
            n.formate(per),
            n.nadh(per),
            n.co2(mit),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
