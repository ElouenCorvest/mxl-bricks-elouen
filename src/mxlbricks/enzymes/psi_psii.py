"""name

EC FIXME

Equilibrator
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
from mxlpy import Derived, Model
from mxlpy.surrogates import qss

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, mass_action_2s, value
from mxlbricks.utils import (
    default_name,
    default_par,
    filter_stoichiometry,
    static,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def _two_div_by(x: float) -> float:
    return 2.0 / x


def _keq_pcp700(
    e0_pc: float,
    f: float,
    eo_p700: float,
    rt: float,
) -> float:
    dg1 = -e0_pc * f
    dg2 = -eo_p700 * f
    dg = -dg1 + dg2
    return math.exp(-dg / rt)


def _keq_faf_d(
    e0_fa: float,
    f: float,
    e0_fd: float,
    rt: float,
) -> float:
    dg1 = -e0_fa * f
    dg2 = -e0_fd * f
    dg = -dg1 + dg2
    return math.exp(-dg / rt)


def _rate_ps1(
    a: float,
    ps2cs: float,
    pfd: float,
) -> float:
    return (1 - ps2cs) * pfd * a


def _rate_ps2(
    b1: float,
    k2: float,
) -> float:
    return 0.5 * k2 * b1


def _ps1states_2019(
    pc_px: float,
    pc_red: float,
    fd_ox: float,
    fd_red: float,
    ps2cs: float,
    psi_tot: float,
    k_fd_red: float,
    keq_fafd: float,
    keq_pcp700: float,
    k_pc_ox: float,
    pfd: float,
) -> float:
    """QSSA calculates open state of PSI
    depends on reduction states of plastocyanin and ferredoxin
    C = [PC], F = [Fd] (ox. forms)
    """
    L = (1 - ps2cs) * pfd
    return psi_tot / (
        1
        + L / (k_fd_red * fd_ox)
        + (1 + fd_red / (keq_fafd * fd_ox))
        * (pc_px / (keq_pcp700 * pc_red) + L / (k_pc_ox * pc_red))
    )


def _ps1states_2021(
    pc_ox: float,
    pc_red: float,
    fd_ox: float,
    fd_red: float,
    ps2cs: float,
    ps1_tot: float,
    k_fd_red: float,
    keq_f: float,
    keq_c: float,
    k_pc_ox: float,
    pfd: float,
    k0: float,
    o2: float,
) -> tuple[float, float, float]:
    """QSSA calculates open state of PSI
    depends on reduction states of plastocyanin and ferredoxin
    C = [PC], F = [Fd] (ox. forms)
    """
    kLI = (1 - ps2cs) * pfd

    y0 = (
        keq_c
        * keq_f
        * pc_red
        * ps1_tot
        * k_pc_ox
        * (fd_ox * k_fd_red + o2 * k0)
        / (
            fd_ox * keq_c * keq_f * pc_red * k_fd_red * k_pc_ox
            + fd_ox * keq_f * k_fd_red * (keq_c * kLI + pc_ox * k_pc_ox)
            + fd_red * k_fd_red * (keq_c * kLI + pc_ox * k_pc_ox)
            + keq_c * keq_f * o2 * pc_red * k0 * k_pc_ox
            + keq_c * keq_f * pc_red * kLI * k_pc_ox
            + keq_f * o2 * k0 * (keq_c * kLI + pc_ox * k_pc_ox)
        )
    )

    y1 = (
        ps1_tot
        * (
            fd_red * k_fd_red * (keq_c * kLI + pc_ox * k_pc_ox)
            + keq_c * keq_f * pc_red * kLI * k_pc_ox
        )
        / (
            fd_ox * keq_c * keq_f * pc_red * k_fd_red * k_pc_ox
            + fd_ox * keq_f * k_fd_red * (keq_c * kLI + pc_ox * k_pc_ox)
            + fd_red * k_fd_red * (keq_c * kLI + pc_ox * k_pc_ox)
            + keq_c * keq_f * o2 * pc_red * k0 * k_pc_ox
            + keq_c * keq_f * pc_red * kLI * k_pc_ox
            + keq_f * o2 * k0 * (keq_c * kLI + pc_ox * k_pc_ox)
        )
    )
    y2 = ps1_tot - y0 - y1

    return y0, y1, y2


def _ps2_crosssection(
    lhc: float,
    static_ant_ii: float,
    static_ant_i: float,
) -> float:
    return static_ant_ii + (1 - static_ant_ii - static_ant_i) * lhc


def _ps2states(
    pq_ox: float,
    pq_red: float,
    ps2cs: float,
    q: float,
    psii_tot: float,
    k2: float,
    k_f: float,
    _kh: float,
    keq_pq_red: float,
    k_pq_red: float,
    pfd: float,
    k_h0: float,
) -> Iterable[float]:
    absorbed = ps2cs * pfd
    kH = k_h0 + _kh * q
    k3p = k_pq_red * pq_ox
    k3m = k_pq_red * pq_red / keq_pq_red

    state_matrix = np.array(
        [
            [-absorbed - k3m, kH + k_f, k3p, 0],
            [absorbed, -(kH + k_f + k2), 0, 0],
            [0, 0, absorbed, -(kH + k_f)],
            [1, 1, 1, 1],
        ],
        dtype=float,
    )
    a = np.array([0, 0, 0, psii_tot])

    return np.linalg.solve(state_matrix, a)


def _b0(
    k_pq_red: float,
    _kh: float,
    pfd: float,
    psii_tot: float,
    k2: float,
    k_h0: float,
    q: float,
    pq_red: float,
    k_f: float,
    pq_ox: float,
    ps2cs: float,
    keq_pq_red: float,
) -> float:
    return (
        k_pq_red
        * keq_pq_red
        * pq_ox
        * psii_tot
        * (
            _kh**2 * q**2
            + _kh * k2 * q
            + 2 * _kh * k_f * q
            + 2 * _kh * k_h0 * q
            + k2 * k_f
            + k2 * k_h0
            + k_f**2
            + 2 * k_f * k_h0
            + k_h0**2
        )
        / (
            _kh**2 * k_pq_red * keq_pq_red * pq_ox * q**2
            + _kh**2 * k_pq_red * pq_red * q**2
            + _kh * k2 * k_pq_red * keq_pq_red * pq_ox * q
            + _kh * k2 * k_pq_red * pq_red * q
            + _kh * k2 * keq_pq_red * pfd * ps2cs * q
            + 2 * _kh * k_f * k_pq_red * keq_pq_red * pq_ox * q
            + 2 * _kh * k_f * k_pq_red * pq_red * q
            + 2 * _kh * k_h0 * k_pq_red * keq_pq_red * pq_ox * q
            + 2 * _kh * k_h0 * k_pq_red * pq_red * q
            + _kh * k_pq_red * keq_pq_red * pfd * pq_ox * ps2cs * q
            + _kh * k_pq_red * pfd * pq_red * ps2cs * q
            + k2 * k_f * k_pq_red * keq_pq_red * pq_ox
            + k2 * k_f * k_pq_red * pq_red
            + k2 * k_f * keq_pq_red * pfd * ps2cs
            + k2 * k_h0 * k_pq_red * keq_pq_red * pq_ox
            + k2 * k_h0 * k_pq_red * pq_red
            + k2 * k_h0 * keq_pq_red * pfd * ps2cs
            + k2 * k_pq_red * pfd * pq_red * ps2cs
            + k2 * keq_pq_red * pfd**2 * ps2cs**2
            + k_f**2 * k_pq_red * keq_pq_red * pq_ox
            + k_f**2 * k_pq_red * pq_red
            + 2 * k_f * k_h0 * k_pq_red * keq_pq_red * pq_ox
            + 2 * k_f * k_h0 * k_pq_red * pq_red
            + k_f * k_pq_red * keq_pq_red * pfd * pq_ox * ps2cs
            + k_f * k_pq_red * pfd * pq_red * ps2cs
            + k_h0**2 * k_pq_red * keq_pq_red * pq_ox
            + k_h0**2 * k_pq_red * pq_red
            + k_h0 * k_pq_red * keq_pq_red * pfd * pq_ox * ps2cs
            + k_h0 * k_pq_red * pfd * pq_red * ps2cs
        )
    )


def _b1(
    k_pq_red: float,
    _kh: float,
    pfd: float,
    psii_tot: float,
    k2: float,
    k_h0: float,
    q: float,
    pq_red: float,
    k_f: float,
    pq_ox: float,
    ps2cs: float,
    keq_pq_red: float,
) -> float:
    return (
        k_pq_red
        * keq_pq_red
        * pfd
        * pq_ox
        * ps2cs
        * psii_tot
        * (_kh * q + k_f + k_h0)
        / (
            _kh**2 * k_pq_red * keq_pq_red * pq_ox * q**2
            + _kh**2 * k_pq_red * pq_red * q**2
            + _kh * k2 * k_pq_red * keq_pq_red * pq_ox * q
            + _kh * k2 * k_pq_red * pq_red * q
            + _kh * k2 * keq_pq_red * pfd * ps2cs * q
            + 2 * _kh * k_f * k_pq_red * keq_pq_red * pq_ox * q
            + 2 * _kh * k_f * k_pq_red * pq_red * q
            + 2 * _kh * k_h0 * k_pq_red * keq_pq_red * pq_ox * q
            + 2 * _kh * k_h0 * k_pq_red * pq_red * q
            + _kh * k_pq_red * keq_pq_red * pfd * pq_ox * ps2cs * q
            + _kh * k_pq_red * pfd * pq_red * ps2cs * q
            + k2 * k_f * k_pq_red * keq_pq_red * pq_ox
            + k2 * k_f * k_pq_red * pq_red
            + k2 * k_f * keq_pq_red * pfd * ps2cs
            + k2 * k_h0 * k_pq_red * keq_pq_red * pq_ox
            + k2 * k_h0 * k_pq_red * pq_red
            + k2 * k_h0 * keq_pq_red * pfd * ps2cs
            + k2 * k_pq_red * pfd * pq_red * ps2cs
            + k2 * keq_pq_red * pfd**2 * ps2cs**2
            + k_f**2 * k_pq_red * keq_pq_red * pq_ox
            + k_f**2 * k_pq_red * pq_red
            + 2 * k_f * k_h0 * k_pq_red * keq_pq_red * pq_ox
            + 2 * k_f * k_h0 * k_pq_red * pq_red
            + k_f * k_pq_red * keq_pq_red * pfd * pq_ox * ps2cs
            + k_f * k_pq_red * pfd * pq_red * ps2cs
            + k_h0**2 * k_pq_red * keq_pq_red * pq_ox
            + k_h0**2 * k_pq_red * pq_red
            + k_h0 * k_pq_red * keq_pq_red * pfd * pq_ox * ps2cs
            + k_h0 * k_pq_red * pfd * pq_red * ps2cs
        )
    )


def _b2(
    k_pq_red: float,
    _kh: float,
    pfd: float,
    psii_tot: float,
    k2: float,
    k_h0: float,
    q: float,
    pq_red: float,
    k_f: float,
    pq_ox: float,
    ps2cs: float,
    keq_pq_red: float,
) -> float:
    return (
        psii_tot
        * (
            _kh**2 * k_pq_red * pq_red * q**2
            + _kh * k2 * k_pq_red * pq_red * q
            + _kh * k2 * keq_pq_red * pfd * ps2cs * q
            + 2 * _kh * k_f * k_pq_red * pq_red * q
            + 2 * _kh * k_h0 * k_pq_red * pq_red * q
            + k2 * k_f * k_pq_red * pq_red
            + k2 * k_f * keq_pq_red * pfd * ps2cs
            + k2 * k_h0 * k_pq_red * pq_red
            + k2 * k_h0 * keq_pq_red * pfd * ps2cs
            + k_f**2 * k_pq_red * pq_red
            + 2 * k_f * k_h0 * k_pq_red * pq_red
            + k_h0**2 * k_pq_red * pq_red
        )
        / (
            _kh**2 * k_pq_red * keq_pq_red * pq_ox * q**2
            + _kh**2 * k_pq_red * pq_red * q**2
            + _kh * k2 * k_pq_red * keq_pq_red * pq_ox * q
            + _kh * k2 * k_pq_red * pq_red * q
            + _kh * k2 * keq_pq_red * pfd * ps2cs * q
            + 2 * _kh * k_f * k_pq_red * keq_pq_red * pq_ox * q
            + 2 * _kh * k_f * k_pq_red * pq_red * q
            + 2 * _kh * k_h0 * k_pq_red * keq_pq_red * pq_ox * q
            + 2 * _kh * k_h0 * k_pq_red * pq_red * q
            + _kh * k_pq_red * keq_pq_red * pfd * pq_ox * ps2cs * q
            + _kh * k_pq_red * pfd * pq_red * ps2cs * q
            + k2 * k_f * k_pq_red * keq_pq_red * pq_ox
            + k2 * k_f * k_pq_red * pq_red
            + k2 * k_f * keq_pq_red * pfd * ps2cs
            + k2 * k_h0 * k_pq_red * keq_pq_red * pq_ox
            + k2 * k_h0 * k_pq_red * pq_red
            + k2 * k_h0 * keq_pq_red * pfd * ps2cs
            + k2 * k_pq_red * pfd * pq_red * ps2cs
            + k2 * keq_pq_red * pfd**2 * ps2cs**2
            + k_f**2 * k_pq_red * keq_pq_red * pq_ox
            + k_f**2 * k_pq_red * pq_red
            + 2 * k_f * k_h0 * k_pq_red * keq_pq_red * pq_ox
            + 2 * k_f * k_h0 * k_pq_red * pq_red
            + k_f * k_pq_red * keq_pq_red * pfd * pq_ox * ps2cs
            + k_f * k_pq_red * pfd * pq_red * ps2cs
            + k_h0**2 * k_pq_red * keq_pq_red * pq_ox
            + k_h0**2 * k_pq_red * pq_red
            + k_h0 * k_pq_red * keq_pq_red * pfd * pq_ox * ps2cs
            + k_h0 * k_pq_red * pfd * pq_red * ps2cs
        )
    )


def _b3(
    k_pq_red: float,
    _kh: float,
    pfd: float,
    psii_tot: float,
    k2: float,
    k_h0: float,
    q: float,
    pq_red: float,
    k_f: float,
    pq_ox: float,
    ps2cs: float,
    keq_pq_red: float,
) -> float:
    return (
        pfd
        * ps2cs
        * psii_tot
        * (
            _kh * k_pq_red * pq_red * q
            + k2 * k_pq_red * pq_red
            + k2 * keq_pq_red * pfd * ps2cs
            + k_f * k_pq_red * pq_red
            + k_h0 * k_pq_red * pq_red
        )
        / (
            _kh**2 * k_pq_red * keq_pq_red * pq_ox * q**2
            + _kh**2 * k_pq_red * pq_red * q**2
            + _kh * k2 * k_pq_red * keq_pq_red * pq_ox * q
            + _kh * k2 * k_pq_red * pq_red * q
            + _kh * k2 * keq_pq_red * pfd * ps2cs * q
            + 2 * _kh * k_f * k_pq_red * keq_pq_red * pq_ox * q
            + 2 * _kh * k_f * k_pq_red * pq_red * q
            + 2 * _kh * k_h0 * k_pq_red * keq_pq_red * pq_ox * q
            + 2 * _kh * k_h0 * k_pq_red * pq_red * q
            + _kh * k_pq_red * keq_pq_red * pfd * pq_ox * ps2cs * q
            + _kh * k_pq_red * pfd * pq_red * ps2cs * q
            + k2 * k_f * k_pq_red * keq_pq_red * pq_ox
            + k2 * k_f * k_pq_red * pq_red
            + k2 * k_f * keq_pq_red * pfd * ps2cs
            + k2 * k_h0 * k_pq_red * keq_pq_red * pq_ox
            + k2 * k_h0 * k_pq_red * pq_red
            + k2 * k_h0 * keq_pq_red * pfd * ps2cs
            + k2 * k_pq_red * pfd * pq_red * ps2cs
            + k2 * keq_pq_red * pfd**2 * ps2cs**2
            + k_f**2 * k_pq_red * keq_pq_red * pq_ox
            + k_f**2 * k_pq_red * pq_red
            + 2 * k_f * k_h0 * k_pq_red * keq_pq_red * pq_ox
            + 2 * k_f * k_h0 * k_pq_red * pq_red
            + k_f * k_pq_red * keq_pq_red * pfd * pq_ox * ps2cs
            + k_f * k_pq_red * pfd * pq_red * ps2cs
            + k_h0**2 * k_pq_red * keq_pq_red * pq_ox
            + k_h0**2 * k_pq_red * pq_red
            + k_h0 * k_pq_red * keq_pq_red * pfd * pq_ox * ps2cs
            + k_h0 * k_pq_red * pfd * pq_red * ps2cs
        )
    )


def add_ps2_cross_section(
    model: Model,
    lhc: str | None = None,
    static_ant_i: str | None = None,
    static_ant_ii: str | None = None,
) -> Model:
    model.add_derived(
        name=n.ps2cs(),
        fn=_ps2_crosssection,
        args=[
            default_name(lhc, n.lhc),
            default_par(model, par=static_ant_ii, name="staticAntII", value=0.1),
            default_par(model, par=static_ant_i, name="staticAntI", value=0.37),
        ],
    )
    return model


def add_psii(
    model: Model,
    *,
    rxn: str,
    pq_ox: str,
    pq_red: str,
    ps2cs: str,
    quencher: str,
    pfd: str,
    h_lumen: str,
    b0: str,
    b1: str,
    b2: str,
    b3: str,
) -> Model:
    model.add_surrogate(
        "ps2states",
        surrogate=qss.Surrogate(
            model=_ps2states,
            args=[
                pq_ox,
                pq_red,
                ps2cs,
                quencher,
                "PSII_total",
                "k2",
                "kF",
                "kH",
                n.keq(pq_red),
                "kPQred",
                pfd,
                "kH0",
            ],
            outputs=[
                b0,
                b1,
                b2,
                b3,
            ],
        ),
    )

    model.add_reaction(
        name=rxn,
        fn=_rate_ps2,
        stoichiometry={
            pq_ox: -1,
            h_lumen: Derived(fn=_two_div_by, args=["bH"]),
        },
        args=[
            b1,
            "k2",
        ],
    )
    return model


def add_psii_analytic(
    model: Model,
    *,
    rxn: str,
    pq_ox: str,
    pq_red: str,
    ps2cs: str,
    quencher: str,
    pfd: str,
    h_lumen: str,
    b0: str,
    b1: str,
    b2: str,
    b3: str,
) -> Model:
    """Use analytically obtain solution for ps2"""

    k_pq_red = "kPQred"
    psii_tot = "PSII_total"
    _kh = "kH"
    k2 = "k2"
    k_h0 = "kH0"
    k_f = "kF"
    keq_pq_red = n.keq(pq_red)
    q = quencher

    model.add_derived(
        b0,
        fn=_b0,
        args=[
            k_pq_red,
            _kh,
            pfd,
            psii_tot,
            k2,
            k_h0,
            q,
            pq_red,
            k_f,
            pq_ox,
            ps2cs,
            keq_pq_red,
        ],
    )
    model.add_derived(
        b1,
        fn=_b1,
        args=[
            k_pq_red,
            _kh,
            pfd,
            psii_tot,
            k2,
            k_h0,
            q,
            pq_red,
            k_f,
            pq_ox,
            ps2cs,
            keq_pq_red,
        ],
    )
    model.add_derived(
        b2,
        fn=_b2,
        args=[
            k_pq_red,
            _kh,
            pfd,
            psii_tot,
            k2,
            k_h0,
            q,
            pq_red,
            k_f,
            pq_ox,
            ps2cs,
            keq_pq_red,
        ],
    )
    model.add_derived(
        b3,
        fn=_b3,
        args=[
            k_pq_red,
            _kh,
            pfd,
            psii_tot,
            k2,
            k_h0,
            q,
            pq_red,
            k_f,
            pq_ox,
            ps2cs,
            keq_pq_red,
        ],
    )

    model.add_reaction(
        name=rxn,
        fn=_rate_ps2,
        stoichiometry={
            pq_ox: -1,
            h_lumen: Derived(fn=_two_div_by, args=["bH"]),
        },
        args=[
            b1,
            "k2",
        ],
    )
    return model


def add_psi_2019(
    model: Model,
    *,
    rxn: str,
    ps2cs: str,
    pfd: str,
    pc_ox: str,
    pc_red: str,
    fd_ox: str,
    fd_red: str,
    a1: str,
    keq_pcp700: str,
    keq_fd_red: str,
) -> Model:
    model.add_derived(
        name=n.a1(),
        fn=_ps1states_2019,
        args=[
            pc_ox,
            pc_red,
            fd_ox,
            fd_red,
            ps2cs,
            "PSI_total",
            "kFdred",
            keq_fd_red,
            keq_pcp700,
            "kPCox",
            pfd,
        ],
    )
    model.add_reaction(
        name=rxn,
        fn=_rate_ps1,
        stoichiometry={
            fd_ox: -1,
            pc_ox: 1,
        },
        args=[
            a1,
            ps2cs,
            pfd,
        ],
    )
    return model


def add_psi_2021(
    model: Model,
    *,
    rxn: str,
    ps2cs: str,
    pfd: str,
    o2_lumen: str,
    pc_ox: str,
    pc_red: str,
    fd_ox: str,
    fd_red: str,
    a0: str,
    a1: str,
    a2: str,
    keq_pcp700: str,
    keq_fd_red: str,
    k_mehler: str,
) -> Model:
    model.add_surrogate(
        "ps1states",
        surrogate=qss.Surrogate(
            model=_ps1states_2021,
            args=[
                pc_ox,
                pc_red,
                fd_ox,
                fd_red,
                ps2cs,
                "PSI_total",
                "kFdred",
                keq_fd_red,
                keq_pcp700,
                "kPCox",
                pfd,
                k_mehler,
                o2_lumen,
            ],
            outputs=[
                a0,
                a1,
                a2,
            ],
        ),
    )
    model.add_reaction(
        name=rxn,
        fn=_rate_ps1,
        stoichiometry={
            pc_ox: 1,
        },
        args=[
            a0,
            ps2cs,
            pfd,
        ],
    )
    return model


def add_mehler(
    model: Model,
    *,
    rxn: str,
    o2_lumen: str,
    h2o2: str,
    a1: str,
    convf: str,
    k_mehler: str,
) -> Model:
    model.add_reaction(
        name=rxn,
        fn=mass_action_2s,
        stoichiometry={
            h2o2: Derived(fn=value, args=[convf]),
        },
        args=[
            a1,
            o2_lumen,
            k_mehler,
        ],
    )
    return model


def add_photosystems(
    model: Model,
    mode: Literal["matrix", "analytical"],
    *,
    rxn_psii: str | None = None,
    rxn_psi: str | None = None,
    rxn_mehler: str | None = None,
    pq_ox: str | None = None,
    pq_red: str | None = None,
    ps2cs: str | None = None,
    quencher: str | None = None,
    pfd: str | None = None,
    o2_lumen: str | None = None,
    h_lumen: str | None = None,
    h2o2: str | None = None,
    pc_ox: str | None = None,
    pc_red: str | None = None,
    fd_ox: str | None = None,
    fd_red: str | None = None,
    a0: str | None = None,
    a1: str | None = None,
    a2: str | None = None,
    b0: str | None = None,
    b1: str | None = None,
    b2: str | None = None,
    b3: str | None = None,
    mehler: bool,
    convf: str | None = None,
) -> Model:
    """PSII: 2 H2O + 2 PQ + 4 H_stroma -> O2 + 2 PQH2 + 4 H_lumen
    PSI: Fd_ox + PC_red -> Fd_red + PC_ox
    """
    pq_ox = default_name(pq_ox, n.pq_ox)
    pq_red = default_name(pq_red, n.pq_red)
    ps2cs = default_name(ps2cs, n.ps2cs)
    quencher = default_name(quencher, n.quencher)
    pfd = default_name(pfd, n.pfd)
    o2_lumen = default_name(h_lumen, lambda: n.o2("_lumen"))
    h_lumen = default_name(h_lumen, lambda: n.h("_lumen"))
    h2o2 = default_name(h2o2, n.h2o2)
    pc_ox = default_name(pc_ox, n.pc_ox)
    pc_red = default_name(pc_red, n.pc_red)
    fd_ox = default_name(fd_ox, n.fd_ox)
    fd_red = default_name(fd_red, n.fd_red)
    a0 = default_name(a0, n.a0)
    a1 = default_name(a1, n.a1)
    a2 = default_name(a2, n.a2)
    b0 = default_name(b0, n.b0)
    b1 = default_name(b1, n.b1)
    b2 = default_name(b2, n.b2)
    b3 = default_name(b3, n.b3)

    model.add_parameter("PSII_total", 2.5)
    model.add_parameter("PSI_total", 2.5)
    model.add_parameter("kH0", 500000000.0)
    model.add_parameter("kPQred", 250.0)
    model.add_parameter("kPCox", 2500.0)
    model.add_parameter("kFdred", 250000.0)
    model.add_parameter("k2", 5000000000.0)
    model.add_parameter("kH", 5000000000.0)
    model.add_parameter("kF", 625000000.0)
    convf = static(model, n.convf(), 3.2e-2) if convf is None else convf

    model.add_derived(
        keq_pcp700 := n.keq("PCP700"),
        _keq_pcp700,
        args=["E^0_PC", "F", "E^0_P700", "RT"],
    )
    model.add_derived(
        keq_fd_red := n.keq(n.ferredoxin_reductase()),
        _keq_faf_d,
        args=["E^0_FA", "F", "E^0_Fd", "RT"],
    )

    if mode == "matrix":
        add_psii(
            model,
            rxn=default_name(rxn_psii, n.ps2),
            pq_ox=pq_ox,
            pq_red=pq_red,
            ps2cs=ps2cs,
            quencher=quencher,
            pfd=pfd,
            h_lumen=h_lumen,
            b0=b0,
            b1=b1,
            b2=b2,
            b3=b3,
        )
    else:
        add_psii_analytic(
            model,
            rxn=default_name(rxn_psii, n.ps2),
            pq_ox=pq_ox,
            pq_red=pq_red,
            ps2cs=ps2cs,
            quencher=quencher,
            pfd=pfd,
            h_lumen=h_lumen,
            b0=b0,
            b1=b1,
            b2=b2,
            b3=b3,
        )

    if not mehler:
        add_psi_2019(
            model,
            rxn=default_name(rxn_psi, n.ps1),
            ps2cs=ps2cs,
            pfd=pfd,
            pc_ox=pc_ox,
            pc_red=pc_red,
            fd_ox=fd_ox,
            fd_red=fd_red,
            a1=a1,
            keq_pcp700=keq_pcp700,
            keq_fd_red=keq_fd_red,
        )
    else:
        model.add_parameter(k_mehler := "kMehler", 1.0)
        add_psi_2021(
            model,
            rxn=default_name(rxn_psi, n.ps1),
            ps2cs=ps2cs,
            pfd=pfd,
            o2_lumen=o2_lumen,
            pc_ox=pc_ox,
            pc_red=pc_red,
            fd_ox=fd_ox,
            fd_red=fd_red,
            a0=a0,
            a1=a1,
            a2=a2,
            keq_pcp700=keq_pcp700,
            keq_fd_red=keq_fd_red,
            k_mehler=k_mehler,
        )
        add_mehler(
            model,
            rxn=default_name(rxn_mehler, n.mehler),
            o2_lumen=o2_lumen,
            h2o2=h2o2,
            a1=a1,
            convf=convf,
            k_mehler=k_mehler,
        )
    return model


def add_energy_production(
    model: Model,
    *,
    rxn: str | None = None,
    energy: str | None = None,
    pfd: str | None = None,
) -> Model:
    rxn = default_name(rxn, n.ps2)
    energy = default_name(energy, n.energy)
    pfd = default_name(pfd, n.pfd)

    model.add_parameter(k := n.kcat(pfd), 1 / 145)  # Fitted
    model.add_parameter(pfd, 700)

    model.add_reaction(
        n.petc(),
        mass_action_1s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                # Substrates
                # Products
                energy: 1,
            },
        ),
        args=[
            pfd,
            k,
        ],
    )
    return model
