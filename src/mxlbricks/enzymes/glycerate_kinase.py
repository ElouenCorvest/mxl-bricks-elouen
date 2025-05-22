"""glycerate kinase

ATP + D-Glycerate <=> ADP + 3-Phospho-D-glycerate

Equilibrator
ATP(aq) + D-Glycerate(aq) ⇌ ADP(aq) + 3-Phospho-D-glycerate(aq)
Keq = 4.9e2 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    reversible_michaelis_menten_2s_2p,
)
from mxlbricks.utils import static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.glycerate_kinase()


def add_glycerate_kinase(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.25) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1) if kmp is None else kmp  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 5.71579) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 490.0) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_2p,
        stoichiometry={
            n.glycerate(): -1.0,
            n.atp(): -1.0,
            n.pga(): 1.0,
        },
        args=[
            n.glycerate(),
            n.atp(),
            n.pga(),
            n.adp(),
            vmax,
            kms,  # FIXME: km_atp missing
            kmp,  # FIXME: ki missing
            keq,
        ],
    )
    return model
