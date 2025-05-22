"""EC 1.2.1.75

Metacyc:
MALONYL-COA_m + NADPH_m + PROTON_m <=> CO-A_m + MALONATE-S-ALD_m + NADP_m

Equilibrator
Malonyl-CoA(aq) + NADPH(aq) ⇌ Malonate semialdehyde(aq) + NADP(aq) + CoA(aq)
Keq = 5.6e-3 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    reversible_michaelis_menten_2s_3p,
)
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.malonyl_coa_reductase()


def add_malonyl_coa_reductase(
    model: Model,
    mit: str,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.03) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1.0) if kmp is None else kmp  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 50.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 0.0056) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_parameter(keq := n.keq(ENZYME), 0.0056)
    model.add_parameter(kmp := n.km(ENZYME, "p"), 1)
    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_3p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.malonyl_coa(mit): -1,
                n.nadph(mit): -1,
                n.malonate_s_aldehyde(mit): 1,
                n.coa(mit): 1,
            },
        ),
        args=[
            n.malonyl_coa(mit),
            n.nadph(mit),
            n.malonate_s_aldehyde(mit),
            n.nadp(mit),
            n.coa(mit),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
