"""SBPase

EC 3.1.3.37

Equilibrator
    H2O(l) + Sedoheptulose 1,7-bisphosphate(aq)
    ⇌ Orthophosphate(aq) + Sedoheptulose 7-phosphate(aq)
    Keq = 2e2 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, michaelis_menten_1s_1i
from mxlbricks.utils import static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.sbpase()


def add_sbpase(
    model: Model,
    *,
    chl_stroma: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    ki: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.013) if kms is None else kms  # FIXME: source
    ki = (
        static(model, n.ki(ENZYME, n.pi()), 12.0) if ki is None else ki
    )  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 0.04 * 8) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source

    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_1s_1i,
        stoichiometry={
            n.sbp(chl_stroma): -1,
            n.s7p(chl_stroma): 1,
        },
        args=[
            n.sbp(chl_stroma),
            n.pi(chl_stroma),
            vmax,
            kms,
            ki,
        ],
    )
    return model
