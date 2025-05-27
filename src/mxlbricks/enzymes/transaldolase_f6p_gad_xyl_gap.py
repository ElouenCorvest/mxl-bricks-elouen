"""F6P + Glycolaldehyde <=> GAP + XYLULOSE

EC 2.2.1.2

Equilibrator
Fructose-6-phosphate(aq) + Glycolaldehyde(aq)
    ⇌ Glyceraldehyde 3-phosphate(aq) + Xylulose(aq)
Keq = 4.8e-4 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, reversible_michaelis_menten_2s_2p
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.transaldolase_f6p_gad_gap_xyl()


def add_transaldolase_f6p_gad_xyl_gap(
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
    keq = static(model, n.keq(ENZYME), 4.8e-4) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_2p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.f6p(compartment): -1.0,
                n.glycolaldehyde(compartment): -1.0,
                n.gap(compartment): 1.0,
                n.xylulose(compartment): 1.0,
            },
        ),
        args=[
            n.f6p(compartment),
            n.glycolaldehyde(compartment),
            n.gap(compartment),
            n.xylulose(compartment),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
