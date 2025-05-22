"""glycolate oxidase

O2(per) + Glycolate(chl) <=> H2O2(per) + Glyoxylate(per)

Equilibrator
O2(aq) + Glycolate(aq) ⇌ H2O2(aq) + Glyoxylate(aq)
Keq = 3e15 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, michaelis_menten_1s, michaelis_menten_2s
from mxlbricks.utils import static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.glycolate_oxidase()


def add_glycolate_oxidase(
    model: Model,
    *,
    compartment: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.06) if kms is None else kms  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 100) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_2s,
        stoichiometry={
            n.glycolate(compartment): -1,
            n.glyoxylate(compartment): 1,
            n.h2o2(compartment): 1,
        },
        args=[
            n.glycolate(compartment),
            n.o2(compartment),
            vmax,
            kms,
        ],
    )
    return model


def add_glycolate_oxidase_yokota(
    model: Model,
    *,
    compartment: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
) -> Model:
    """

    This variant doesn't actually include the oxygen concentration
    """
    kms = static(model, n.kms(ENZYME), 0.06) if kms is None else kms  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 100) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_1s,
        stoichiometry={
            n.glycolate(compartment): -1,
            n.glyoxylate(compartment): 1,
            n.h2o2(compartment): 1,
        },
        args=[
            n.glycolate(compartment),
            vmax,
            kms,
        ],
    )
    return model
