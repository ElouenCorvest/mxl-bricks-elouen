"""serine glyoxylate transaminase

Glyoxylate + L-Serine <=> Glycine + Hydroxypyruvate

EC 2.6.1.45

Equilibrator
Glyoxylate(aq) + Serine(aq) ⇌ Glycine(aq) + Hydroxypyruvate(aq)
Keq = 6 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    ping_pong_bi_bi,
    reversible_michaelis_menten_2s_2p,
)
from mxlbricks.utils import static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.serine_glyoxylate_transaminase()


def add_serine_glyoxylate_transaminase_irreversible(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    km_gox: str | None = None,
    km_ser: str | None = None,
) -> Model:
    km_gox = (
        static(model, n.km(ENZYME, n.glyoxylate()), 0.15) if km_gox is None else km_gox
    )  # FIXME: source
    km_ser = (
        static(model, n.km(ENZYME, n.serine()), 2.72) if km_ser is None else km_ser
    )  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 159.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    stoichiometry = {
        n.glyoxylate(): -1.0,
        n.serine(): -1.0,
        n.glycine(): 1.0,
        n.hydroxypyruvate(): 1.0,
    }

    model.add_reaction(
        name=ENZYME,
        fn=ping_pong_bi_bi,
        stoichiometry=stoichiometry,
        args=[
            n.glyoxylate(),
            n.serine(),
            vmax,
            km_gox,
            km_ser,
        ],
    )

    return model


def add_serine_glyoxylate_transaminase(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    km_gox: str | None = None,
    km_ser: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    km_gox = (
        static(model, n.km(ENZYME, n.glyoxylate()), 0.15) if km_gox is None else km_gox
    )  # FIXME: source
    km_ser = (
        static(model, n.km(ENZYME, n.serine()), 2.72) if km_ser is None else km_ser
    )  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1)
    keq = static(model, n.keq(ENZYME), 6)
    kcat = (
        static(model, n.kcat(ENZYME), 159.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_2p,
        stoichiometry={
            n.glyoxylate(): -1.0,
            n.serine(): -1.0,
            n.glycine(): 1.0,
            n.hydroxypyruvate(): 1.0,
        },
        args=[
            n.glyoxylate(),
            n.serine(),
            n.glycine(),
            n.hydroxypyruvate(),
            vmax,
            km_gox,  # FIXME: kms2 missing
            kmp,
            keq,
        ],
    )
    return model
