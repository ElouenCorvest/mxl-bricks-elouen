"""name

EC 2.7.9.1

Equilibrator
Pyruvate(aq) + ATP(aq) + Orthophosphate(aq) ⇌ PEP(aq) + AMP(aq) + Diphosphate(aq)
Keq = 9.6e-3 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, reversible_michaelis_menten_3s_3p
from mxlbricks.utils import static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.pyruvate_phosphate_dikinase()


def add_pyruvate_phosphate_dikinase(
    model: Model,
    chl_stroma: str = "",
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
    keq = static(model, n.keq(ENZYME), 0.0096) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_3s_3p,
        stoichiometry={
            n.pyruvate(chl_stroma): -1,
            n.atp(chl_stroma): -1,
            n.pep(chl_stroma): 1,
        },
        args=[
            n.pyruvate(chl_stroma),
            n.atp(chl_stroma),
            n.pi(chl_stroma),
            n.pep(chl_stroma),
            n.amp(chl_stroma),
            n.ppi(chl_stroma),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
