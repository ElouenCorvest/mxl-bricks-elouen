"""EC 1.5.1.5

Metacyc: METHYLENETHFDEHYDROG-NADP-RXN
METHENYL-THF_m + NADPH_m + 0.93 PROTON_m <=> METHYLENE-THF_m + NADP_m

Equilibrator
5,10-Methenyltetrahydrofolate(aq) + NADPH(aq) ⇌ 5,10-Methylenetetrahydrofolate(aq) + NADP(aq)
Keq = 1e1 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    reversible_michaelis_menten_2s_2p,
)
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.methylene_thf_dehydrogenase()


def add_methylene_thf_dehydrogenase(
    model: Model,
    mit: str,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.12) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 0.302) if kmp is None else kmp  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 14.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 10.0) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_2p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.methenyl_thf(mit): -1,
                n.nadph(mit): -1,
                n.methylene_thf(mit): 1,
                n.nadp(mit): 1,
            },
        ),
        args=[
            n.methenyl_thf(mit),
            n.nadph(mit),
            n.methylene_thf(mit),
            n.nadp(mit),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )

    return model
