"""Lumped reaction of Glyceraldehyde 3-phosphate dehydrogenase (GADPH) and Phosphoglycerate kinase (PGK)
    3-Phospho-D-glycerate(aq) + ATP(aq) ⇌ 3-Phospho-D-glyceroyl phosphate(aq) + ADP(aq)
    3-Phospho-D-glyceroyl phosphate(aq) + NADPH(aq) ⇌ D-Glyceraldehyde 3-phosphate(aq) + NADP (aq) + Orthophosphate(aq)
Into
    3-Phospho-D-glycerate(aq) + ATP(aq) + NADPH(aq) ⇌ D-Glyceraldehyde 3-phosphate(aq) + ADP(aq) + Orthophosphate(aq) + NADP(aq)

Equilibrator
    Keq = 6.0e-4 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, reversible_michaelis_menten_3s_4p
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = "pgk_gadph"


def lumped_pgk_gadph(
    model: Model,
    *,
    compartment: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.18) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 0.27) if kmp is None else kmp  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 537) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 6.0e-4) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_3s_4p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.pga(compartment): -1.0,
                n.atp(compartment): -1.0,
                n.nadph(compartment): -1.0,
                n.gap(compartment): 1.0,
                n.adp(compartment): 1.0,
                n.pi(compartment): 1.0,
                n.nadp(compartment): 1.0,
            },
        ),
        args=[
            n.pga(compartment),
            n.atp(compartment),
            n.nadph(compartment),
            n.gap(compartment),
            n.adp(compartment),
            n.pi(compartment),
            n.nadp(compartment),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
