"""acetoacetate_coa_ligase

EC 6.2.1.16

Metacyc (ACETOACETATE--COA-LIGASE-RXN):
    3-KETOBUTYRATE_m + ATP_m + CO-A_m
    --> ACETOACETYL-COA_m + AMP_m + Diphosphate_m + 0.92 PROTON_m

Equilibrator
    Acetoacetate(aq) + ATP(aq) + CoA(aq) ⇌ Acetoacetyl-CoA(aq) + AMP(aq) + Diphosphate(aq)
    Keq = 2 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    reversible_michaelis_menten_3s_3p,
)
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.acetoacetate_coa_ligase()


def add_acetoacetate_coa_ligase(
    model: Model,
    compartment: str = "",
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.07) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1) if kmp is None else kmp  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 5.89) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 2) if keq is None else keq
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    stoichiometry = filter_stoichiometry(
        model,
        {
            n.acetoacetate(compartment): -1,
            n.atp(compartment): -1,
            n.coa(compartment): -1,
            n.acetoacetyl_coa(compartment): 1,
            n.amp(compartment): 1,
            n.ppi(compartment): 1,
        },
    )

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_3s_3p,
        stoichiometry=stoichiometry,
        args=[
            n.acetoacetate(compartment),
            n.atp(compartment),
            n.coa(compartment),
            n.acetoacetyl_coa(compartment),
            n.amp(compartment),
            n.ppi(compartment),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )

    return model
