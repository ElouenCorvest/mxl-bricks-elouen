"""name

EC 6.4.1.2

Metacyc:
ACETYL-COA_m + ATP_m + HCO3_m <=> ADP_m + MALONYL-COA_m + PROTON_m + Pi_m

Equilibrator
Acetyl-CoA(aq) + ATP(aq) + HCO3-(aq) ⇌ ADP(aq) + Malonyl-CoA(aq) + Orthophosphate(aq)
Too much uncertainty for HCO3

As a proxy
Acetyl-CoA(aq) + ATP(aq) + CO2(total) ⇌ ADP(aq) + Malonyl-CoA(aq) + Orthophosphate(aq)
Keq = 4e1 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    reversible_michaelis_menten_3s_3p,
    reversible_michaelis_menten_3s_3p_1i,
)
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model

ENZYME = n.acetyl_coa_carboxyltransfer()


def add_acetyl_coa_carboxyltransfer(
    model: Model,
    compartment: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.0487) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 0.1) if kmp is None else kmp  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 30.1) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 40.0) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    stoichiometry = filter_stoichiometry(
        model,
        stoichiometry={
            n.acetyl_coa(compartment): -1.0,
            n.atp(compartment): -1.0,
            n.hco3(compartment): -1.0,
            n.adp(compartment): 1.0,
            n.malonyl_coa(compartment): 1.0,
            n.pi(compartment): 1.0,
        },
    )

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_3s_3p,
        stoichiometry=stoichiometry,
        args=[
            n.acetyl_coa(compartment),
            n.atp(compartment),
            n.hco3(compartment),
            n.adp(compartment),
            n.malonyl_coa(compartment),
            n.pi(compartment),
            vmax,
            kms,
            kmp,
            keq,
        ],
    )

    return model


def add_acetyl_coa_carboxyltransfer_1i(
    model: Model,
    compartment: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
    ki: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.0487) if kms is None else kms  # FIXME: source
    kmp = static(model, n.kmp(ENZYME), 0.1) if kmp is None else kmp  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 30.1) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), 40.0) if keq is None else keq  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])
    ki = static(model, n.ki(ENZYME), 0.002)  # FIXME: source

    stoichiometry = filter_stoichiometry(
        model,
        stoichiometry={
            n.acetyl_coa(compartment): -1.0,
            n.atp(compartment): -1.0,
            n.hco3(compartment): -1.0,
            n.adp(compartment): 1.0,
            n.malonyl_coa(compartment): 1.0,
            n.pi(compartment): 1.0,
        },
    )

    model.add_reaction(
        name=ENZYME,
        fn=reversible_michaelis_menten_3s_3p_1i,
        stoichiometry=stoichiometry,
        args=[
            n.acetyl_coa(compartment),
            n.atp(compartment),
            n.hco3(compartment),
            n.adp(compartment),
            n.malonyl_coa(compartment),
            n.pi(compartment),
            vmax,
            kms,
            kmp,
            keq,
            n.formate(),
            ki,
        ],
    )
    return model
