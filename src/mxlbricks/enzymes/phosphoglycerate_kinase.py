"""Phosphoglycerate kinase (PGK)

EC 2.7.2.3

kcat
    - 537 | 1 /s | Pseudomonas sp. | brenda

km
    - 0.18 | PGA | mM | Synechocystis sp. | brenda
    - ? | BPGA | mM | Synechocystis sp. | brenda
    - 0.3 | ATP | mM | Spinacia oleracea | brenda
    - 0.27 | ADP | mM | Spinacia oleracea | brenda


Equilibrator
    ATP(aq) + 3-Phospho-D-glycerate(aq) ⇌ ADP(aq) + 3-Phospho-D-glyceroyl phosphate(aq)
    Keq = 3.7e-4 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    rapid_equilibrium_2s_2p,
    reversible_michaelis_menten_2s_2p,
)
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.phosphoglycerate_kinase()


def add_phosphoglycerate_kinase_poolman(
    model: Model,
    *,
    compartment: str = "",
    kre: str | None = None,
    keq: str | None = None,
) -> Model:
    kre = static(model, n.kre(ENZYME), 800000000.0) if kre is None else kre
    keq = static(model, n.keq(ENZYME), 0.00031) if keq is None else keq

    model.add_reaction(
        name=ENZYME,
        fn=rapid_equilibrium_2s_2p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.pga(compartment): -1.0,
                n.atp(compartment): -1.0,
                n.bpga(compartment): 1.0,
                n.adp(compartment): 1.0,
            },
        ),
        args=[
            n.pga(compartment),
            n.atp(compartment),
            n.bpga(compartment),
            n.adp(compartment),
            kre,
            keq,
        ],
    )
    return model


def add_phosphoglycerate_kinase(
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
    keq = static(model, n.keq(ENZYME), 3.7e-4) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_2p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.pga(compartment): -1.0,
                n.atp(compartment): -1.0,
                n.bpga(compartment): 1.0,
                n.adp(compartment): 1.0,
            },
        ),
        args=[
            n.pga(compartment),
            n.atp(compartment),
            n.bpga(compartment),
            n.adp(compartment),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
