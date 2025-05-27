"""fructose-1,6-bisphosphatase

EC 3.1.3.11

Equilibrator

Equilibrator
    H2O(l) + D-Fructose 1,6-bisphosphate(aq) ⇌ Orthophosphate(aq) + D-Fructose 6-phosphate(aq)
    Keq = 1.2e3 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, michaelis_menten_1s_2i
from mxlbricks.utils import static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.fbpase()


def add_fbpase(
    model: Model,
    *,
    chl_stroma: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    ki_f6p: str | None = None,
    ki_pi: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.03) if kms is None else kms  # FIXME: source
    ki_f6p = (
        static(model, n.ki(ENZYME, n.f6p()), 0.7) if ki_f6p is None else ki_f6p
    )  # FIXME: source
    ki_pi = (
        static(model, n.ki(ENZYME, n.pi()), 12.0) if ki_pi is None else ki_pi
    )  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 0.2 * 8) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_1s_2i,
        stoichiometry={
            n.fbp(chl_stroma): -1,
            n.f6p(chl_stroma): 1,
        },
        args=[
            n.fbp(chl_stroma),
            n.f6p(chl_stroma),
            n.pi(chl_stroma),
            vmax,
            kms,
            ki_f6p,
            ki_pi,
        ],
    )
    return model
