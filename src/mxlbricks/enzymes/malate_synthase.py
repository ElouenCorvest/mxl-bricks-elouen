"""malate synthase

EC 2.3.3.9

Equilibrator
------------
    H2O + Acetyl-CoA + Glyoxylate <=> CoA + (S)-Malate
    Keq = 6.0e6 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, reversible_michaelis_menten_2s_2p
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.malate_synthase()


def add_malate_synthase(
    model: Model,
    chl_stroma: str = "",
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.098) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1.0) if kmp is None else kmp  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 27.8) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 6.0e6) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_2p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.acetyl_coa(chl_stroma): -1,
                n.glyoxylate(chl_stroma): -1,
                n.coa(chl_stroma): 1,
                n.malate(chl_stroma): 1,
            },
        ),
        args=[
            n.acetyl_coa(chl_stroma),
            n.glyoxylate(chl_stroma),
            n.coa(chl_stroma),
            n.malate(chl_stroma),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
