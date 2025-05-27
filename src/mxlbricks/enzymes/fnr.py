r"""Ferredoxin-NADP reductase

2 reduced ferredoxin + NADP+ + H+ ⇌ \rightleftharpoons 2 oxidized ferredoxin + NADPH

EC 1.18.1.2

Equilibrator
"""

import math

from mxlpy import Derived, Model

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    michaelis_menten_1s,
    michaelis_menten_2s,
    value,
)
from mxlbricks.utils import filter_stoichiometry, static

ENZYME = n.fnr()


def _keq_fnr(
    E0_Fd: float,
    F: float,
    E0_NADP: float,
    pHstroma: float,
    dG_pH: float,
    RT: float,
) -> float:
    dg1 = -E0_Fd * F
    dg2 = -2 * E0_NADP * F
    dg = -2 * dg1 + dg2 + dG_pH * pHstroma
    return math.exp(-dg / RT)


def _rate_fnr2016(
    Fd_ox: float,
    Fd_red: float,
    NADPH: float,
    NADP: float,
    KM_FNR_F: float,
    KM_FNR_N: float,
    vmax: float,
    Keq_FNR: float,
) -> float:
    fdred = Fd_red / KM_FNR_F
    fdox = Fd_ox / KM_FNR_F
    nadph = NADPH / KM_FNR_N
    nadp = NADP / KM_FNR_N
    return (
        vmax
        * (fdred**2 * nadp - fdox**2 * nadph / Keq_FNR)
        / ((1 + fdred + fdred**2) * (1 + nadp) + (1 + fdox + fdox**2) * (1 + nadph) - 1)
    )


def _rate_fnr_2019(
    Fd_ox: float,
    Fd_red: float,
    NADPH: float,
    NADP: float,
    KM_FNR_F: float,
    KM_FNR_N: float,
    vmax: float,
    Keq_FNR: float,
    convf: float,
) -> float:
    fdred = Fd_red / KM_FNR_F
    fdox = Fd_ox / KM_FNR_F
    nadph = NADPH / convf / KM_FNR_N
    nadp = NADP / convf / KM_FNR_N
    return (
        vmax
        * (fdred**2 * nadp - fdox**2 * nadph / Keq_FNR)
        / ((1 + fdred + fdred**2) * (1 + nadp) + (1 + fdox + fdox**2) * (1 + nadph) - 1)
    )


def add_fnr_mmol_chl(
    model: Model,
    *,
    chl_stroma: str = "",
    kcat: str | None = None,
    km_fd: str | None = None,
    km_nadp: str | None = None,
    e0: str | None = None,
) -> Model:
    km_fd = (
        static(model, n.km(ENZYME, n.fd_red()), 1.56) if km_fd is None else km_fd
    )  # FIXME: source
    km_nadp = (
        static(model, n.km(ENZYME, n.nadp()), 0.22) if km_nadp is None else km_nadp
    )  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 500.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 3.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_derived(
        keq := n.keq(ENZYME),
        fn=_keq_fnr,
        args=[
            "E^0_Fd",
            "F",
            "E^0_NADP",
            n.ph(chl_stroma),
            "dG_pH",
            "RT",
        ],
    )

    model.add_reaction(
        name=ENZYME,
        fn=_rate_fnr2016,
        stoichiometry={
            n.fd_ox(): 2,
        },
        args=[
            n.fd_ox(),
            n.fd_red(),
            n.nadph(),
            n.nadp(),
            km_fd,
            km_nadp,
            vmax,
            keq,
        ],
    )

    return model


def add_fnr_mm(
    model: Model,
    *,
    chl_stroma: str = "",
    kcat: str | None = None,
    km_fd: str | None = None,
    km_nadp: str | None = None,
    e0: str | None = None,
    convf: str | None = None,
) -> Model:
    km_fd = (
        static(model, n.km(ENZYME, n.fd_red()), 1.56) if km_fd is None else km_fd
    )  # FIXME: source
    km_nadp = (
        static(model, n.km(ENZYME, n.nadp()), 0.22) if km_nadp is None else km_nadp
    )  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 500.0) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 3.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])
    convf = static(model, n.convf(), 3.2e-2) if convf is None else convf

    model.add_derived(
        n.keq(ENZYME),
        fn=_keq_fnr,
        args=[
            "E^0_Fd",
            "F",
            "E^0_NADP",
            n.ph(chl_stroma),
            "dG_pH",
            "RT",
        ],
    )

    model.add_reaction(
        name=ENZYME,
        fn=_rate_fnr_2019,
        stoichiometry={
            n.fd_ox(): 2,
            n.nadph(): Derived(fn=value, args=[convf]),
        },
        args=[
            n.fd_ox(),
            n.fd_red(),
            n.nadph(),
            n.nadp(),
            km_fd,
            km_nadp,
            vmax,
            n.keq(ENZYME),
            convf,
        ],
    )
    return model


def add_fnr_static(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
) -> Model:
    """Saadat version to put into Poolman model"""
    kms = static(model, n.kms(ENZYME), 0.19) if kms is None else kms  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 2.816) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source

    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_1s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.nadp(): -1.0,
                n.nadph(): 1.0,
            },
        ),
        args=[
            n.nadp(),
            vmax,
            kms,
        ],
    )

    return model


def add_fnr_energy_dependent(
    model: Model,
    *,
    compartment: str = "",
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
) -> Model:
    kms = static(model, n.kms(ENZYME), 0.19) if kms is None else kms  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 2.816) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=michaelis_menten_2s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                # Substrates
                n.nadp(compartment): -1.0,
                n.energy(compartment): -1.0,
                # Products
                n.nadph(compartment): 1.0,
            },
        ),
        args=[
            n.nadp(),
            n.energy(),
            vmax,
            kms,
        ],
    )

    return model
