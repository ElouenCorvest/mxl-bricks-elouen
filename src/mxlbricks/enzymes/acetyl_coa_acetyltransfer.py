"""name

EC 2.3.1.9

Metacyc:
ACETOACETYL-COA_m + CO-A_m  <=>  2.0 ACETYL-COA_m

Equilibrator
Acetoacetyl-CoA(aq) + CoA(aq) ⇌ 2 Acetyl-CoA(aq)
Keq = 2.4e4 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    reversible_michaelis_menten_2s_1p,
)
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.acetyl_coa_acetyltransfer()


def add_acetyl_coa_acetyltransfer(
    model: Model,
    *,
    compartment: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.0176) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 0.1386) if kmp is None else kmp  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 220.5) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 24000.0) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    stoichiometry = filter_stoichiometry(
        model,
        stoichiometry={
            n.acetoacetyl_coa(compartment): -1,
            n.coa(compartment): -1,
            n.acetyl_coa(compartment): 2,
        },
    )

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_1p,
        stoichiometry=stoichiometry,
        args=[
            n.acetoacetyl_coa(compartment),
            n.coa(compartment),
            n.acetyl_coa(compartment),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )

    return model
