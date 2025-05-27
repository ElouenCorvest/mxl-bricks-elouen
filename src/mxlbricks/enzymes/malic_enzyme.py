"""malic enzyme == malate dehydrogenase decarboxylating

EC 1.1.1.39

Equilibrator
    NAD (aq) + (S)-Malate(aq) + H2O(l) ⇌ NADH(aq) + Pyruvate(aq) + CO2(total)
    Keq = 0.2 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, reversible_michaelis_menten_2s_3p
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.malic_enzyme()


def add_malic_enzyme(
    model: Model,
    chl_stroma: str = "",
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.003) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 0.00125) if kmp is None else kmp  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 39) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 0.2) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_3p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.nad(chl_stroma): -1,
                n.malate(chl_stroma): -1,
                n.nadh(chl_stroma): 1,
                n.pyruvate(chl_stroma): 1,
                n.co2(chl_stroma): 1,
            },
        ),
        args=[
            n.nad(chl_stroma),
            n.malate(chl_stroma),
            n.nadh(chl_stroma),
            n.pyruvate(chl_stroma),
            n.co2(chl_stroma),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
