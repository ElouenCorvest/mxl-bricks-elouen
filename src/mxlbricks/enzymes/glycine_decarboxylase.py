"""glycine decarboxylase

2 Glycine + NAD + 2 H2O ⇌ Serine + NH3 + NADH + CO2

Equilibrator
2 Glycine(aq) + NAD(aq) + 2 H2O(l) ⇌ Serine(aq) + NH3(aq) + NADH(aq) + CO2(total)
Keq = 2.4e-4 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    michaelis_menten_1s,
    michaelis_menten_2s,
    reversible_michaelis_menten_2s_4p,
)
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.glycine_decarboxylase()


def add_glycine_decarboxylase_yokota(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 6.0) if kms is None else kms  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 100.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_1s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.glycine(): -2.0,
                n.serine(): 1.0,
            },
        ),
        args=[
            n.glycine(),
            vmax,
            kms,
        ],
    )
    return model


def add_glycine_decarboxylase_irreversible(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 6.0) if kms is None else kms  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 100.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_2s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.glycine(): -2.0,
                n.nad(): -1.0,
                n.serine(): 1.0,
                n.nh4(): 1.0,
                n.nadh(): 1.0,
                n.co2(): 1.0,
            },
        ),
        args=[
            n.glycine(),
            n.nad(),
            vmax,
            kms,
        ],
    )

    return model


def add_glycine_decarboxylase(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 6.0) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 1.0) if kmp is None else kmp  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 100.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 0.00024) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_2s_4p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.glycine(): -2.0,
                n.nad(): -1.0,
                n.serine(): 1.0,
                n.nh4(): 1.0,
                n.nadh(): 1.0,
                n.co2(): 1.0,
            },
        ),
        args=[
            n.glycine(),
            n.nad(),
            n.serine(),
            n.nh4(),
            n.nadh(),
            n.co2(),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )
    return model
