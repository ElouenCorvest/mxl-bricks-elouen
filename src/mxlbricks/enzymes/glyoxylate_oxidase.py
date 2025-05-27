"""name

EC 1.2.3.5

Equilibrator
Glyoxylate(aq) + H2O(l) + O2(aq) ⇌ Oxalate(aq) + H2O2(aq)
Keq = 2.5e28 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, michaelis_menten_2s
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.glyoxylate_oxidase()


def add_glyoxylate_oxidase(
    model: Model,
    *,
    compartment: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 1.0) if kms is None else kms  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 1.0) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_2s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.glyoxylate(compartment): -1.0,
                # n.h2o(compartment): -1.0,
                n.o2(compartment): -1.0,
                n.oxalate(compartment): 1.0,
                n.h2o2(compartment): 1.0,
            },
        ),
        args=[
            n.glyoxylate(compartment),
            n.o2(compartment),
            vmax,
            kms,
        ],
    )
    return model
