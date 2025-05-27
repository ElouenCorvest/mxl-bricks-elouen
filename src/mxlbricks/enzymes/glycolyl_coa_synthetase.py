"""GCS: atp + coa + glyclt -> Diphosphate + amp + glyccoa

dG' = 9.25
Keq = 0.024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    michaelis_menten_3s,
    reversible_michaelis_menten_3s_3p,
)
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.glycolyl_coa_synthetase()


def add_glycolyl_coa_synthetase_irrev(
    model: Model,
    *,
    chl_stroma: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 13) if kms is None else kms  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 4.0) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_3s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.atp(chl_stroma): -1,
                n.coa(chl_stroma): -1,
                n.glycolate(chl_stroma): -1,
                n.glycolyl_coa(chl_stroma): 1,
                n.ppi(chl_stroma): 1,
                n.amp(chl_stroma): 1,
            },
        ),
        args=[
            n.atp(chl_stroma),
            n.coa(chl_stroma),
            n.glycolate(chl_stroma),
            vmax,
            kms,
        ],
    )
    return model


def add_glycolyl_coa_synthetase(
    model: Model,
    *,
    chl_stroma: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 13) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1.0) if kmp is None else kmp  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 4.0) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 0.024) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_3s_3p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.atp(chl_stroma): -1,
                n.coa(chl_stroma): -1,
                n.glycolate(chl_stroma): -1,
                n.glycolyl_coa(chl_stroma): 1,
                n.ppi(chl_stroma): 1,
                n.amp(chl_stroma): 1,
            },
        ),
        args=[
            n.atp(chl_stroma),
            n.coa(chl_stroma),
            n.glycolate(chl_stroma),
            n.glycolyl_coa(chl_stroma),
            n.ppi(chl_stroma),
            n.amp(chl_stroma),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
