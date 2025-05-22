"""name

EC 6.2.1.5

Metacyc: SUCCCOASYN-RXN
SUC_m + CO-A_m + ATP_m <=> SUC-COA_m + ADP_m + Pi_m

Equilibrator
Succinate(aq) + CoA(aq) + ATP(aq) ⇌ Succinyl-CoA(aq) + ADP(aq) + Pi(aq)
Keq = 2 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    reversible_michaelis_menten_2s_3p,
)
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.succinyl_coa_synthetase()


def add_succinyl_coa_synthetase(
    model: Model,
    mit: str,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.25) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 0.041) if kmp is None else kmp  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 44.73) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 2) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_3p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.succinate(mit): -1,
                n.coa(mit): -1,
                n.succinyl_coa(mit): 1,
                n.adp(mit): 1,
                n.pi(mit): 1,
            },
        ),
        args=[
            n.succinate(mit),
            n.coa(mit),
            n.succinyl_coa(mit),
            n.adp(mit),
            n.pi(mit),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )

    return model
