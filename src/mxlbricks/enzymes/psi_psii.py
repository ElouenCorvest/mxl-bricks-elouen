"""name

EC FIXME

Equilibrator
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mxlpy import Derived, Model
from mxlpy.surrogates import qss

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s, mass_action_2s, value
from mxlbricks.utils import filter_stoichiometry, static

if TYPE_CHECKING:
    from collections.abc import Iterable


def _two_div_by(x: float) -> float:
    return 2.0 / x


def _keq_pcp700(
    E0_PC: float,
    F: float,
    E0_P700: float,
    RT: float,
) -> float:
    DG1 = -E0_PC * F
    DG2 = -E0_P700 * F
    DG = -DG1 + DG2
    K: float = np.exp(-DG / RT)
    return K


def _keq_faf_d(
    E0_FA: float,
    F: float,
    E0_Fd: float,
    RT: float,
) -> float:
    DG1 = -E0_FA * F
    DG2 = -E0_Fd * F
    DG = -DG1 + DG2
    K: float = np.exp(-DG / RT)
    return K


def _rate_ps1(
    A: float,
    ps2cs: float,
    pfd: float,
) -> float:
    return (1 - ps2cs) * pfd * A


def _rate_ps2(
    B1: float,
    k2: float,
) -> float:
    return 0.5 * k2 * B1


def _ps1states_2019(
    PC: float,
    PCred: float,
    Fd: float,
    Fdred: float,
    ps2cs: float,
    PSItot: float,
    kFdred: float,
    Keq_FAFd: float,
    Keq_PCP700: float,
    kPCox: float,
    pfd: float,
) -> float:
    """QSSA calculates open state of PSI
    depends on reduction states of plastocyanin and ferredoxin
    C = [PC], F = [Fd] (ox. forms)
    """
    L = (1 - ps2cs) * pfd
    return PSItot / (
        1
        + L / (kFdred * Fd)
        + (1 + Fdred / (Keq_FAFd * Fd))
        * (PC / (Keq_PCP700 * PCred) + L / (kPCox * PCred))
    )


def _ps1states_2021(
    PC: float,
    PCred: float,
    Fd: float,
    Fdred: float,
    ps2cs: float,
    PSItot: float,
    kFdred: float,
    KeqF: float,
    KeqC: float,
    kPCox: float,
    pfd: float,
    k0: float,
    O2: float,
) -> tuple[float, float, float]:
    """QSSA calculates open state of PSI
    depends on reduction states of plastocyanin and ferredoxin
    C = [PC], F = [Fd] (ox. forms)
    """
    kLI = (1 - ps2cs) * pfd

    y0 = (
        KeqC
        * KeqF
        * PCred
        * PSItot
        * kPCox
        * (Fd * kFdred + O2 * k0)
        / (
            Fd * KeqC * KeqF * PCred * kFdred * kPCox
            + Fd * KeqF * kFdred * (KeqC * kLI + PC * kPCox)
            + Fdred * kFdred * (KeqC * kLI + PC * kPCox)
            + KeqC * KeqF * O2 * PCred * k0 * kPCox
            + KeqC * KeqF * PCred * kLI * kPCox
            + KeqF * O2 * k0 * (KeqC * kLI + PC * kPCox)
        )
    )

    y1 = (
        PSItot
        * (
            Fdred * kFdred * (KeqC * kLI + PC * kPCox)
            + KeqC * KeqF * PCred * kLI * kPCox
        )
        / (
            Fd * KeqC * KeqF * PCred * kFdred * kPCox
            + Fd * KeqF * kFdred * (KeqC * kLI + PC * kPCox)
            + Fdred * kFdred * (KeqC * kLI + PC * kPCox)
            + KeqC * KeqF * O2 * PCred * k0 * kPCox
            + KeqC * KeqF * PCred * kLI * kPCox
            + KeqF * O2 * k0 * (KeqC * kLI + PC * kPCox)
        )
    )
    y2 = PSItot - y0 - y1

    return y0, y1, y2


def _ps2_crosssection(
    LHC: float,
    staticAntII: float,
    staticAntI: float,
) -> float:
    return staticAntII + (1 - staticAntII - staticAntI) * LHC


def _ps2states(
    PQ: float,
    PQred: float,
    ps2cs: float,
    Q: float,
    PSIItot: float,
    k2: float,
    kF: float,
    _kH: float,
    Keq_PQred: float,
    kPQred: float,
    pfd: float,
    kH0: float,
) -> Iterable[float]:
    absorbed = ps2cs * pfd
    kH = kH0 + _kH * Q
    k3p = kPQred * PQ
    k3m = kPQred * PQred / Keq_PQred

    state_matrix = np.array(
        [
            [-absorbed - k3m, kH + kF, k3p, 0],
            [absorbed, -(kH + kF + k2), 0, 0],
            [0, 0, absorbed, -(kH + kF)],
            [1, 1, 1, 1],
        ],
        dtype=float,
    )
    a = np.array([0, 0, 0, PSIItot])

    return np.linalg.solve(state_matrix, a)


def add_ps2_cross_section(
    model: Model,
    static_ant_i: str | None = None,
    static_ant_ii: str | None = None,
) -> Model:
    static_ant_i = (
        static(model, "staticAntI", 0.37) if static_ant_i is None else static_ant_i
    )
    static_ant_ii = (
        static(model, "staticAntII", 0.1) if static_ant_ii is None else static_ant_ii
    )

    model.add_derived(
        name=n.ps2cs(),
        fn=_ps2_crosssection,
        args=[
            n.lhc(),
            static_ant_ii,
            static_ant_i,
        ],
    )
    return model


def add_photosystems(
    model: Model,
    *,
    chl_lumen: str,
    mehler: bool,
    convf: str | None = None,
) -> Model:
    """PSII: 2 H2O + 2 PQ + 4 H_stroma -> O2 + 2 PQH2 + 4 H_lumen
    PSI: Fd_ox + PC_red -> Fd_red + PC_ox
    """
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
        n.keq("PCP700"),
        _keq_pcp700,
        args=["E^0_PC", "F", "E^0_P700", "RT"],
    )
    model.add_derived(
        n.keq(n.ferredoxin_reductase()),
        _keq_faf_d,
        args=["E^0_FA", "F", "E^0_Fd", "RT"],
    )

    model.add_surrogate(
        "ps2states",
        surrogate=qss.Surrogate(
            model=_ps2states,
            args=[
                n.pq_ox(),
                n.pq_red(),
                n.ps2cs(),
                n.quencher(),
                "PSII_total",
                "k2",
                "kF",
                "kH",
                n.keq(n.pq_red()),
                "kPQred",
                n.pfd(),
                "kH0",
            ],
            outputs=[
                n.b0(),
                n.b1(),
                n.b2(),
                n.b3(),
            ],
        ),
    )

    enzyme_name = n.ps2()
    model.add_reaction(
        name=enzyme_name,
        fn=_rate_ps2,
        stoichiometry={
            n.pq_ox(): -1,
            n.h(chl_lumen): Derived(fn=_two_div_by, args=["bH"]),
        },
        args=[
            n.b1(),
            "k2",
        ],
    )

    enzyme_name = n.ps1()
    if not mehler:
        model.add_derived(
            name=n.a1(),
            fn=_ps1states_2019,
            args=[
                n.pc_ox(),
                n.pc_red(),
                n.fd_ox(),
                n.fd_red(),
                n.ps2cs(),
                "PSI_total",
                "kFdred",
                n.keq(n.ferredoxin_reductase()),
                n.keq("PCP700"),
                "kPCox",
                n.pfd(),
            ],
        )
        model.add_reaction(
            name=enzyme_name,
            fn=_rate_ps1,
            stoichiometry={
                n.fd_ox(): -1,
                n.pc_ox(): 1,
            },
            args=[
                n.a1(),
                n.ps2cs(),
                n.pfd(),
            ],
        )
    else:
        model.add_parameter("kMehler", 1.0)

        model.add_surrogate(
            "ps1states",
            surrogate=qss.Surrogate(
                model=_ps1states_2021,
                args=[
                    n.pc_ox(),
                    n.pc_red(),
                    n.fd_ox(),
                    n.fd_red(),
                    n.ps2cs(),
                    "PSI_total",
                    "kFdred",
                    n.keq(n.ferredoxin_reductase()),
                    n.keq("PCP700"),
                    "kPCox",
                    n.pfd(),
                    "kMehler",
                    n.o2(chl_lumen),
                ],
                outputs=[
                    n.a0(),
                    n.a1(),
                    n.a2(),
                ],
            ),
        )

        model.add_reaction(
            name=enzyme_name,
            fn=_rate_ps1,
            stoichiometry={
                n.pc_ox(): 1,
            },
            args=[n.a0(), n.ps2cs(), n.pfd()],
        )
        model.add_reaction(
            name=n.mehler(),
            fn=mass_action_2s,
            stoichiometry={n.h2o2(): Derived(fn=value, args=[convf])},
            args=[
                n.a1(),
                n.o2(chl_lumen),
                "kMehler",
            ],
        )
    return model


def add_energy_production(model: Model) -> Model:
    model.add_parameter(k := n.kcat(n.pfd()), 1 / 145)  # Fitted
    model.add_parameter(n.pfd(), 700)

    model.add_reaction(
        n.petc(),
        mass_action_1s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                # Substrates
                # Products
                n.energy(): 1,
            },
        ),
        args=[n.pfd(), k],
    )
    return model
