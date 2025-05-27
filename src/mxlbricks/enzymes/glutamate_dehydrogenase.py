"""EC 1.4.1.3

Equilibrator
NADPH(aq) + NH3(aq) + 2-Oxoglutarate(aq) ⇌ H2O(l) + NADP(aq) + L-Glutamate(aq)
Keq = 7.2e5 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, reversible_michaelis_menten_3s_2p
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.glutamate_dehydrogenase()


def add_glutamate_dehydrogenase(
    model: Model,
    *,
    compartment: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 1.54) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 0.64) if kmp is None else kmp  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 104) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 7.2e5) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_3s_2p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.nadph(compartment): -1.0,
                n.nh4(compartment): -1.0,
                n.oxoglutarate(compartment): -1.0,
                n.glutamate(compartment): 1.0,
                n.nadp(compartment): 1.0,
            },
        ),
        args=[
            n.nadph(compartment),
            n.nh4(compartment),
            n.oxoglutarate(compartment),
            n.glutamate(compartment),
            n.nadp(compartment),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
