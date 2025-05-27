"""TCR1: nadph + tarcoa -> nadp + coa + 2h3oppan

Tartronyl-Coa + NADPH -> Tartronate-semialdehyde + NADP + CoA
dG' = 29.78
Keq = 6.06e-6
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, reversible_michaelis_menten_2s_3p
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.tartronyl_coa_reductase()


def add_tartronyl_coa_reductase(
    model: Model,
    *,
    chl_stroma: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.03) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1.0) if kmp is None else kmp  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 1.4) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = (
        static(model, n.keq(ENZYME), 6.06e-06) if keq is None else keq
    )  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_3p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.nadph(chl_stroma): -1,
                n.tartronyl_coa(chl_stroma): -1,
                n.nadp(chl_stroma): 1,
                n.tartronate_semialdehyde(chl_stroma): 1,
            },
        ),
        args=[
            # substrates
            n.tartronyl_coa(chl_stroma),
            n.nadph(chl_stroma),
            # products
            n.tartronate_semialdehyde(chl_stroma),
            n.nadp(chl_stroma),
            n.coa(chl_stroma),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
