"""H+-transporting two-sector ATPase

ADP + Orthophosphate -> ATP

EC 3.6.3.14

Equilibrator
ADP(aq) + Orthophosphate(aq) ⇌ ATP(aq) + H2O(l)
Keq = 6.4e-6 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from mxlpy import Derived, Model

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, neg_div, value
from mxlbricks.utils import filter_stoichiometry, static

ENZYME = n.atp_synthase()


def _keq_atp(
    pH: float,
    DeltaG0_ATP: float,
    dG_pH: float,
    HPR: float,
    pHstroma: float,
    Pi_mol: float,
    RT: float,
) -> float:
    delta_g = DeltaG0_ATP - dG_pH * HPR * (pHstroma - pH)
    return Pi_mol * math.exp(-delta_g / RT)


def _rate_atp_synthase_2000(
    adp: float,
    pi: float,
    v16: float,
    km161: float,
    km162: float,
) -> float:
    return v16 * adp * pi / ((adp + km161) * (pi + km162))


def _rate_atp_synthase_2016(
    ATP: float,
    ADP: float,
    Keq_ATPsynthase: float,
    kATPsynth: float,
) -> float:
    return kATPsynth * (ADP - ATP / Keq_ATPsynthase)


def _rate_atp_synthase_2019(
    ATP: float,
    ADP: float,
    Keq_ATPsynthase: float,
    kATPsynth: float,
    convf: float,
) -> float:
    return kATPsynth * (ADP / convf - ATP / convf / Keq_ATPsynthase)


def add_atp_synthase_mmol_chl(
    model: Model,
    *,
    chl_stroma: str = "",
    chl_lumen: str = "_lumen",
    kf: str | None = None,
    hpr: str | None = None,
    bh: str | None = None,  # proton buffering
) -> Model:
    kf = static(model, n.kf(ENZYME), 20.0) if kf is None else kf
    hpr = static(model, "HPR", 14.0 / 3.0) if hpr is None else hpr
    bh = static(model, "bH", 100.0) if bh is None else bh

    model.add_parameter("Pi_mol", 0.01)
    model.add_parameter("DeltaG0_ATP", 30.6)

    model.add_derived(
        name=n.keq(ENZYME),
        fn=_keq_atp,
        args=[
            n.ph(chl_lumen),
            "DeltaG0_ATP",
            "dG_pH",
            hpr,
            n.ph(chl_stroma),
            "Pi_mol",
            "RT",
        ],
    )

    model.add_reaction(
        name=ENZYME,
        fn=_rate_atp_synthase_2016,
        stoichiometry={
            n.atp(chl_stroma): 1.0,
            n.h(chl_lumen): Derived(
                fn=neg_div,
                args=[
                    "HPR",
                    bh,
                ],
            ),
        },
        args=[
            n.atp(chl_stroma),
            n.adp(chl_stroma),
            n.keq(ENZYME),
            kf,
        ],
    )
    return model


def add_atp_synthase_mm(
    model: Model,
    *,
    chl_stroma: str = "",
    chl_lumen: str = "_lumen",
    kf: str | None = None,
    hpr: str | None = None,
    bh: str | None = None,  # proton buffering
    convf: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 20.0) if kf is None else kf
    hpr = static(model, "HPR", 14.0 / 3.0) if hpr is None else hpr
    bh = static(model, "bH", 100.0) if bh is None else bh
    convf = static(model, n.convf(), 3.2e-2) if convf is None else convf

    model.add_parameter("Pi_mol", 0.01)
    model.add_parameter("DeltaG0_ATP", 30.6)

    model.add_derived(
        name=n.keq(ENZYME),
        fn=_keq_atp,
        args=[
            n.ph(chl_lumen),
            "DeltaG0_ATP",
            "dG_pH",
            hpr,
            n.ph(chl_stroma),
            "Pi_mol",
            "RT",
        ],
    )

    model.add_reaction(
        name=ENZYME,
        fn=_rate_atp_synthase_2019,
        stoichiometry={
            n.h(chl_lumen): Derived(
                fn=neg_div,
                args=[hpr, bh],
            ),
            n.atp(chl_stroma): Derived(fn=value, args=[convf]),
        },
        args=[
            n.atp(chl_stroma),
            n.adp(chl_stroma),
            n.keq(ENZYME),
            kf,
            convf,
        ],
    )
    return model


def add_atp_synthase_static_protons(
    model: Model,
    *,
    chl_stroma: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    km_adp: str | None = None,
    km_pi: str | None = None,
) -> Model:
    """Used by Poolman 2000"""
    km_adp = (
        static(model, n.km(ENZYME, n.adp()), 0.014) if km_adp is None else km_adp
    )  # FIXME: source
    km_pi = (
        static(model, n.km(ENZYME, n.pi()), 0.3) if km_pi is None else km_pi
    )  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 2.8) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=_rate_atp_synthase_2000,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.adp(chl_stroma): -1.0,
                n.atp(chl_stroma): 1.0,
            },
        ),
        args=[
            n.adp(chl_stroma),
            n.pi(chl_stroma),
            vmax,
            km_adp,
            km_pi,
        ],
    )
    return model


def _rate_static_energy(
    adp: float,
    pi: float,
    energy: float,
    v16: float,
    km161: float,
    km162: float,
) -> float:
    return adp * pi * energy * v16 / ((adp + km161) * (pi + km162))


def add_atp_synthase_energy_dependent(
    model: Model,
    *,
    chl_stroma: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    km_adp: str | None = None,
    km_pi: str | None = None,
) -> Model:
    km_adp = (
        static(model, n.km(ENZYME, n.adp()), 0.014) if km_adp is None else km_adp
    )  # FIXME: source
    km_pi = (
        static(model, n.km(ENZYME, n.pi()), 0.3) if km_pi is None else km_pi
    )  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 2.8) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=_rate_static_energy,
        stoichiometry=filter_stoichiometry(
            model,
            {
                # Substrates
                n.adp(chl_stroma): -1.0,
                n.energy(chl_stroma): -1.0,
                # Products
                n.atp(chl_stroma): 1.0,
            },
        ),
        args=[
            n.adp(chl_stroma),
            n.pi(chl_stroma),
            n.energy(chl_stroma),
            vmax,
            km_adp,
            km_pi,
        ],
    )
    return model


def _atp_gamma(
    Pi: float,
    ATP: float,
    ADP: float,
    convf: float,
) -> float:
    return (ATP / convf) / ((ADP / convf) * (Pi / 1000))


def _delta_g_atp_synthase(
    pH: float,
    gammaATP: float,
    DeltaG0_ATP: float,
    dG_pH: float,
    HPR: float,
    pHstroma: float,
    RT: float,
) -> float:
    return cast(
        float,
        DeltaG0_ATP - dG_pH * HPR * (pHstroma - pH) + RT * np.log(gammaATP),
    )


def _atp_synthase2(
    DeltaGATPsyn: float,
    ATPturnover: float,
) -> float:
    return -DeltaGATPsyn * ATPturnover


def add_atp_synthase_2024(
    model: Model,
    *,
    chl_lumen: str,
    chl_stroma: str = "",
    kf: str | None = None,
    hpr: str | None = None,
    bh: str | None = None,  # proton buffering
    convf: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 20.0) if kf is None else kf
    hpr = static(model, "HPR", 14.0 / 3.0) if hpr is None else hpr
    bh = static(model, "bH", 100.0) if bh is None else bh
    convf = static(model, n.convf(), 3.2e-2) if convf is None else convf

    model.add_parameter(delta_g := "DeltaG0_ATP", 30.6)
    model.add_parameter(atp_turnover := "ATPturnover", 90)

    model.add_derived(
        name="ATP_gamma",
        fn=_atp_gamma,
        args=[
            n.pi(),
            n.atp(),
            n.adp(),
            convf,
        ],
    )

    model.add_derived(
        name="DeltaGATPsyn",
        fn=_delta_g_atp_synthase,
        args=[
            n.ph(chl_lumen),
            "ATP_gamma",
            delta_g,
            "dG_pH",
            hpr,
            n.ph(chl_stroma),
            "RT",
        ],
    )

    model.add_reaction(
        name="vATPsynthase",
        fn=_atp_synthase2,
        stoichiometry={
            n.h(chl_lumen): Derived(fn=neg_div, args=[hpr, bh]),
            n.atp(chl_stroma): Derived(fn=value, args=[convf]),
        },
        args=[
            "DeltaGATPsyn",
            atp_turnover,
        ],
    )
    return model
