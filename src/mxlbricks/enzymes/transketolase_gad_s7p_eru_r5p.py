"""Glycolaldehyde + S7P <=> RIBOSE_5P  + ERYTHRULOSE

EC 2.2.1.1

Equilibrator
Glycolaldehyde(aq) + Sedoheptulose-7-phosphate(aq)
    ⇌ Ribose-5-phosphate(aq) + Erythrulose(aq)
Keq = 0.5 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, reversible_michaelis_menten_2s_2p
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.transketolase_gad_s7p_r5p_eru()


def add_transketolase_gad_s7p_eru_r5p(
    model: Model,
    compartment: str = "",
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.1) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1.0) if kmp is None else kmp  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 1.0) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 0.5) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_2p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.glycolaldehyde(compartment): -1.0,
                n.s7p(compartment): -1.0,
                n.r5p(compartment): 1.0,
                n.erythrulose(compartment): 1.0,
            },
        ),
        args=[
            n.glycolaldehyde(compartment),
            n.s7p(compartment),
            n.r5p(compartment),
            n.erythrulose(compartment),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
