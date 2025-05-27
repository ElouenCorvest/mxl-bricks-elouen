"""R1P + ATP  <=> RUBP + ADP

EC FIXME

Equilibrator
Ribose-1-phosphate(aq) + ATP(aq) ⇌ Ribulose-1,5-bisphosphate(aq) + ADP(aq)
Keq = 4.4e6 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, michaelis_menten_2s
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.r1p_kinase()


def add_r1p_kinase(
    model: Model,
    compartment: str = "",
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.1) if kms is None else kms  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 1.0) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_2s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.r1p(compartment): -1.0,
                n.atp(compartment): -1.0,
                n.rubp(compartment): 1.0,
                n.adp(compartment): 1.0,
            },
        ),
        args=[
            n.r1p(compartment),
            n.atp(compartment),
            vmax,
            kms,
        ],
    )
    return model
