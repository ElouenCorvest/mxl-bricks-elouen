"""name

EC 1.1.1.37

Equilibrator
Oxaloacetate(aq) + NADH(aq) ⇌ Malate(aq) + NAD(aq)
Keq = 4.4e4 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, reversible_michaelis_menten_2s_2p
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.malate_dehydrogenase()


def add_malate_dehydrogenase(
    model: Model,
    chl_stroma: str = "",
    per: str = "",
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.1) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1.0) if kmp is None else kmp  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 1.0) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 44000.0) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_2p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.oxaloacetate(per): -1,
                n.nadh(per): -1,
                n.malate(chl_stroma): 1,
                n.nad(per): 1,
            },
        ),
        args=[
            n.oxaloacetate(per),
            n.nadh(per),
            n.malate(chl_stroma),
            n.nad(chl_stroma),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
