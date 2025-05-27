"""Ribulose-1,5-bisphosphate carboxylase/oxygenase

Enzyme catalysing both carboxylation as well as oxygenation of ribulose-1,5-bisphosphate
leading to either 2xPGA or 1xPGA and 1xPGO


Equilibrator (carboxylation)
    D-Ribulose 1,5-bisphosphate(aq) + CO2(total) ⇌ 2 3-Phospho-D-glycerate(aq)
    Keq = 1.6e4 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)

Equilibrator (oxygenation)
    Oxygen(aq) + D-Ribulose 1,5-bisphosphate(aq) ⇌ 3-Phospho-D-glycerate(aq) + 2-Phosphoglycolate(aq)
    Keq = 2.9e91 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)


Following inhibition mechanisms are known
    - PGA (Poolman 2000)
    - FBP (Poolman 2000)
    - SBP (Poolman 2000)
    - Orthophosphate (Poolman 2000)
    - NADPH (Poolman 2000)
    - PGO (FIXME)


Because of it's complex dynamics, multiple kinetic descriptions of rubisco are possible,
some of which have been implemented here.
    - Poolman 2000, doi: FIXME
    - Witzel 2010, doi: FIXME

Kinetic parameters
------------------
kcat (CO2)
    - 3 s^1 (Stitt 2010)

Witzel:
    gamma = 1 / km_co2
    omega = 1 / km_o2
    lr = k_er_minus / k_er_plus
    lc = k_er_minus / (omega * kcat_carb)
    lrc = k_er_minus / (gamma * k_er_plus)
    lro = k_er_minus / (omega * k_er_plus)
    lo = k_er_minus / (omega * k_oxy)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import div, mass_action_1s, mul, one_div
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from mxlpy import Model


def _rate_poolman_0i(
    rubp: float, co2: float, vmax: float, kms_rubp: float, kms_co2: float
) -> float:
    return vmax * rubp * co2 / ((rubp + kms_rubp) * (co2 + kms_co2))


def _rate_poolman_1i() -> float:
    raise NotImplementedError


def _rate_poolman_2i() -> float:
    raise NotImplementedError


def _rate_poolman_3i() -> float:
    raise NotImplementedError


def _rate_poolman_4i() -> float:
    raise NotImplementedError


def _rate_poolman_5i(
    rubp: float,
    pga: float,
    co2: float,
    vmax: float,
    kms_rubp: float,
    kms_co2: float,
    # inhibitors
    ki_pga: float,
    fbp: float,
    ki_fbp: float,
    sbp: float,
    ki_sbp: float,
    pi: float,
    ki_p: float,
    nadph: float,
    ki_nadph: float,
) -> float:
    top = vmax * rubp * co2
    btm = (
        rubp
        + kms_rubp
        * (
            1
            + pga / ki_pga
            + fbp / ki_fbp
            + sbp / ki_sbp
            + pi / ki_p
            + nadph / ki_nadph
        )
    ) * (co2 + kms_co2)
    return top / btm


def _rate_witzel_1i() -> float:
    raise NotImplementedError


def _rate_witzel_2i() -> float:
    raise NotImplementedError


def _rate_witzel_3i() -> float:
    raise NotImplementedError


def _rate_witzel_4i() -> float:
    raise NotImplementedError


def _rate_witzel_5i(
    rubp: float,
    s2: float,
    vmax: float,
    gamma_or_omega: float,
    co2: float,
    o2: float,
    lr: float,
    lc: float,
    lo: float,
    lrc: float,
    lro: float,
    i1: float,  # pga
    ki1: float,
    i2: float,  # fbp
    ki2: float,
    i3: float,  # sbp
    ki3: float,
    i4: float,  # pi
    ki4: float,
    i5: float,  # nadph
    ki5: float,
) -> float:
    vmax_app = (gamma_or_omega * vmax * s2 / lr) / (1 / lr + co2 / lrc + o2 / lro)
    km_app = 1 / (1 / lr + co2 / lrc + o2 / lro)
    return (vmax_app * rubp) / (
        rubp
        + km_app
        * (
            1
            + co2 / lc
            + o2 / lo
            + i1 / ki1
            + i2 / ki2
            + i3 / ki3
            + i4 / ki4
            + i5 / ki5
        )
    )


def add_rubisco_poolman(
    model: Model,
    *,
    chl_stroma: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    km_co2: str | None = None,
    km_rubp: str | None = None,
    ki_pga: str | None = None,
    ki_fbp: str | None = None,
    ki_sbp: str | None = None,
    ki_pi: str | None = None,
    ki_nadph: str | None = None,
) -> Model:
    ENZYME = n.rubisco_carboxylase()

    km_co2 = (
        static(model, n.km(ENZYME, n.co2()), 0.0107) if km_co2 is None else km_co2
    )  # FIXME: source
    km_rubp = (
        static(model, n.km(ENZYME, n.rubp()), 0.02) if km_rubp is None else km_rubp
    )  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 0.34 * 8) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    ki_pga = static(model, n.ki(ENZYME, n.pga()), 0.04) if ki_pga is None else ki_pga
    ki_fbp = static(model, n.ki(ENZYME, n.fbp()), 0.04) if ki_fbp is None else ki_fbp
    ki_sbp = static(model, n.ki(ENZYME, n.sbp()), 0.075) if ki_sbp is None else ki_sbp
    ki_pi = static(model, n.ki(ENZYME, n.pi()), 0.9) if ki_pi is None else ki_pi
    ki_nadph = (
        static(model, n.ki(ENZYME, n.nadph()), 0.07) if ki_nadph is None else ki_nadph
    )

    model.add_reaction(
        name=n.rubisco_carboxylase(),
        fn=_rate_poolman_5i,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.rubp(chl_stroma): -1.0,
                n.pga(chl_stroma): 2.0,
                n.co2(chl_stroma): -1,
            },
        ),
        args=[
            n.rubp(chl_stroma),
            n.pga(chl_stroma),
            n.co2(chl_stroma),
            vmax,
            km_rubp,
            km_co2,
            ki_pga,
            n.fbp(chl_stroma),
            ki_fbp,
            n.sbp(chl_stroma),
            ki_sbp,
            n.pi(chl_stroma),
            ki_pi,
            n.nadph(chl_stroma),
            ki_nadph,
        ],
    )

    return model


def add_rubisco(
    model: Model,
    *,
    chl_stroma: str = "",
    kcat_carb: str | None = None,
    kcat_ox: str | None = None,
    e0: str | None = None,
    km_co2: str | None = None,
    km_o2: str | None = None,
    km_rubp: str | None = None,
    k_er_plus: str | None = None,
    k_er_minus: str | None = None,
    ki_pga: str | None = None,
    ki_fbp: str | None = None,
    ki_sbp: str | None = None,
    ki_pi: str | None = None,
    ki_nadph: str | None = None,
) -> Model:
    ENZYME = n.rubisco()

    km_co2 = (
        static(model, n.km(ENZYME, n.co2()), 10.7 / 1000) if km_co2 is None else km_co2
    )  # FIXME: source
    km_o2 = (
        static(model, n.km(ENZYME, n.o2()), 295 / 1000) if km_o2 is None else km_o2
    )  # FIXME: source

    km_rubp = (
        static(model, n.km(ENZYME, n.rubp()), 0.02) if km_rubp is None else km_rubp
    )  # FIXME: source
    kcat_carb = (
        static(model, n.kcat(ENZYME), 3.1) if kcat_carb is None else kcat_carb
    )  # FIXME: source
    kcat_ox = (
        static(model, n.kcat(ENZYME), 1.125) if kcat_carb is None else kcat_carb
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 0.16) if e0 is None else e0  # FIXME: source

    model.add_derived(
        vmax_carb := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat_carb, e0]
    )
    model.add_derived(vmax_ox := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat_ox, e0])

    ki_pga = static(model, n.ki(ENZYME, n.pga()), 0.04) if ki_pga is None else ki_pga
    ki_fbp = static(model, n.ki(ENZYME, n.fbp()), 0.04) if ki_fbp is None else ki_fbp
    ki_sbp = static(model, n.ki(ENZYME, n.sbp()), 0.075) if ki_sbp is None else ki_sbp
    ki_pi = static(model, n.ki(ENZYME, n.pi()), 0.9) if ki_pi is None else ki_pi
    ki_nadph = (
        static(model, n.ki(ENZYME, n.nadph()), 0.07) if ki_nadph is None else ki_nadph
    )

    k_er_plus = (
        static(model, "k_er_plus", 0.15 * 1000) if k_er_plus is None else k_er_plus
    )  # 1 / (mM * s)
    k_er_minus = (
        static(model, "k_er_minus", 0.0048) if k_er_minus is None else k_er_minus
    )  # 1 / s

    model.add_derived(gamma := "gamma", one_div, args=[km_co2])
    model.add_derived(omega := "omega", one_div, args=[km_o2])
    model.add_derived(
        omega_kcat_carb := "omega_kcat_carb",
        mul,
        args=[omega, n.kcat(n.rubisco_carboxylase())],
    )
    model.add_derived(
        omega_koxy := "omega_koxy", mul, args=[omega, n.kcat(n.rubisco_oxygenase())]
    )
    model.add_derived(omega_ker_plus := "omega_ker_plus", mul, args=[omega, k_er_plus])
    model.add_derived(gamma_ker_plus := "gamma_ker_plus", mul, args=[gamma, k_er_plus])
    model.add_derived(lr := "lr", div, args=[k_er_minus, k_er_plus])
    model.add_derived(lc := "lc", div, args=[k_er_minus, omega_kcat_carb])
    model.add_derived(lrc := "lrc", div, args=[k_er_minus, gamma_ker_plus])
    model.add_derived(lro := "lro", div, args=[k_er_minus, omega_ker_plus])
    model.add_derived(lo := "lo", div, args=[k_er_minus, omega_koxy])
    model.add_reaction(
        name=n.rubisco_carboxylase(),
        fn=_rate_witzel_5i,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.rubp(chl_stroma): -1.0,
                n.pga(chl_stroma): 2.0,
                n.co2(chl_stroma): -1,
            },
        ),
        args=[
            n.rubp(),
            n.co2(chl_stroma),
            vmax_carb,
            gamma,  # 1 / km_co2
            n.co2(chl_stroma),
            n.o2(chl_stroma),
            lr,
            lc,
            lo,
            lrc,
            lro,
            n.pga(),
            ki_pga,
            n.fbp(),
            ki_fbp,
            n.sbp(),
            ki_sbp,
            n.pi(),
            ki_pi,
            n.nadph(),
            ki_nadph,
        ],
    )
    model.add_reaction(
        name=n.rubisco_oxygenase(),
        fn=_rate_witzel_5i,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.rubp(): -1.0,
                n.o2(): -1.0,
                n.pga(): 1.0,
                n.pgo(): 1.0,
            },
        ),
        args=[
            n.rubp(chl_stroma),
            n.o2(chl_stroma),
            vmax_ox,
            omega,  # 1 / km_o2
            n.co2(chl_stroma),
            n.o2(chl_stroma),
            lr,
            lc,
            lo,
            lrc,
            lro,
            n.pga(),
            ki_pga,
            n.fbp(),
            ki_fbp,
            n.sbp(),
            ki_sbp,
            n.pi(),
            ki_pi,
            n.nadph(),
            ki_nadph,
        ],
    )

    return model
