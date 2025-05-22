"""glycine transaminase

EC 2.6.1.4

Equilibrator
L-Glutamate(aq) + Glyoxylate(aq) ⇌ 2-Oxoglutarate(aq) + Glycine(aq)
Keq = 30 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    michaelis_menten_1s,
    michaelis_menten_2s,
    reversible_michaelis_menten_2s_2p,
)
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.glycine_transaminase()


def add_glycine_transaminase_yokota(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
) -> Model:
    """Yokota 1980 used reduced stoichiometry for the reaction."""
    kms = static(model, n.kms(ENZYME), 3.0) if kms is None else kms  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 143.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_1s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.glyoxylate(): -1.0,
                n.glycine(): 1.0,
            },
        ),
        args=[
            n.glyoxylate(),
            vmax,
            kms,
        ],
    )
    return model


def add_glycine_transaminase_irreversible(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 3.0) if kms is None else kms  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 143.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_2s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.glutamate(): -1.0,
                n.glyoxylate(): -1.0,
                n.oxoglutarate(): 1.0,
                n.glycine(): 1.0,
            },
        ),
        args=[
            n.glyoxylate(),
            n.glutamate(),
            vmax,
            kms,
        ],
    )

    return model


def add_glycine_transaminase(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 3.0) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1.0) if kmp is None else kmp  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 143.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 30) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_2p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.glutamate(): -1.0,
                n.glyoxylate(): -1.0,
                n.oxoglutarate(): 1.0,
                n.glycine(): 1.0,
            },
        ),
        args=[
            n.glyoxylate(),
            n.glutamate(),
            n.glycine(),
            n.oxoglutarate(),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )

    return model
