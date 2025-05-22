"""EC 3.1.2.10

Metacyc:
FORMYL_COA_m + WATER_m <=> CO-A_m + FORMATE_m + PROTON_m

Equilibrator
Formyl-CoA(aq) + Water(l) ⇌ CoA(aq) + Formate(aq)
Keq = 4e1 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    reversible_michaelis_menten_1s_2p,
)
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.thioesterase()


def add_thioesterase(
    model: Model,
    mit: str,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.045) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1.0) if kmp is None else kmp  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 1.0) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 40.0) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_1s_2p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.formyl_coa(mit): -1,
                n.coa(mit): 1,
                n.formate(mit): 1,
            },
        ),
        args=[
            n.formyl_coa(mit),
            n.coa(mit),
            n.formate(mit),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )

    return model
