"""EC 1.1.99.14

Equilibrator
------------
Glycolate(aq) + NAD (aq) ⇌ Glyoxylate(aq) + NADH(aq)
Keq = 6.2 x 10-8 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, reversible_michaelis_menten_2s_2p
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.glycolate_dehydrogenase()


def add_glycolate_dehydrogenase(
    model: Model,
    chl_stroma: str = "",
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.21) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1.0) if kmp is None else kmp  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 1.93) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 6.2e-8) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_2p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.glycolate(chl_stroma): -1,
                n.nad(): -1,
                n.glyoxylate(chl_stroma): 1,
                n.nadh(): 1,
            },
        ),
        args=[
            n.glycolate(chl_stroma),
            n.nad(),
            n.glyoxylate(chl_stroma),
            n.nadh(),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
