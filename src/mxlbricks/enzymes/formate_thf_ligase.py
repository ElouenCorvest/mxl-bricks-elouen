"""name

EC 6.3.4.3
    Metacyc: FORMATETHFLIG-RXN
    FORMATE_m + ATP_m + THF_m <=> 10-FORMYL-THF_m + ADP_m + Pi_m

    Equilibrator
    Formate(aq) + THF(aq) + ATP(aq) ⇌ 10-Formyltetrahydrofolate(aq) + ADP(aq) + Pi(aq)
    Keq = 2.0 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    reversible_michaelis_menten_3s_3p,
)
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.formate_thf_ligase()


def add_formate_thf_ligase(
    model: Model,
    compartment: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 7.6) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 10.0) if kmp is None else kmp  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 6.08) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 2.0) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_3s_3p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.formate(compartment): -1,
                n.atp(compartment): -1,
                n.thf(compartment): -1,
                n.formyl_thf(compartment): 1,
                n.adp(compartment): 1,
                n.pi(compartment): 1,
            },
        ),
        args=[
            n.formate(compartment),
            n.atp(compartment),
            n.thf(compartment),
            n.formyl_thf(compartment),
            n.adp(compartment),
            n.pi(compartment),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )

    return model
