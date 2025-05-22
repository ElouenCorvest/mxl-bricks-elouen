"""catalase

2 H2O2 <=> 2 H2O + O2

Equilibrator
2 H2O2(aq) ⇌ 2 H2O(l) + O2(aq)
Keq = 4.3e33 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, michaelis_menten_1s
from mxlbricks.utils import static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.catalase()


def add_catalase(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 137.9) if kms is None else kms  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 760500.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_1s,
        stoichiometry={
            n.h2o2(): -1,
        },
        args=[
            n.h2o2(),
            vmax,
            kms,
        ],
    )
    return model
