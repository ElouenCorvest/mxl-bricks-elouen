from typing import cast

import numpy as np
from mxlpy import Derived, Model

from mxlbricks import names as n
from mxlbricks.utils import static

ENZYME = n.b6f()


def _four_div_by(x: float) -> float:
    return 4.0 / x


def _keq_cytb6f(
    pH: float,
    F: float,
    E0_PQ: float,
    E0_PC: float,
    pHstroma: float,
    RT: float,
    dG_pH: float,
) -> float:
    DG1 = -2 * F * E0_PQ
    DG2 = -F * E0_PC
    DG = -(DG1 + 2 * dG_pH * pH) + 2 * DG2 + 2 * dG_pH * (pHstroma - pH)
    Keq = np.exp(-DG / RT)
    return cast(float, Keq)


def _b6f(
    PC_ox: float,
    PQ_ox: float,
    PQ_red: float,
    PC_red: float,
    Keq_B6f: float,
    kCytb6f: float,
) -> float:
    return cast(
        float,
        np.maximum(
            kCytb6f * (PQ_red * PC_ox**2 - PQ_ox * PC_red**2 / Keq_B6f),
            -kCytb6f,
        ),
    )


def add_b6f(
    model: Model,
    *,
    chl_stroma: str = "",
    chl_lumen: str,
    bh: str | None = None,
) -> Model:
    bh = static(model, "bH", 100.0) if bh is None else bh

    model.add_parameter(n.kcat(ENZYME), 2.5)
    model.add_derived(
        name=n.keq(ENZYME),
        fn=_keq_cytb6f,
        args=[
            n.ph(chl_lumen),
            "F",
            "E^0_PQ",
            "E^0_PC",
            n.ph(chl_stroma),
            "RT",
            "dG_pH",
        ],
    )
    model.add_reaction(
        name=ENZYME,
        fn=_b6f,
        stoichiometry={
            n.pc_ox(): -2,
            n.pq_ox(): 1,
            n.h(chl_lumen): Derived(fn=_four_div_by, args=[bh]),
        },
        args=[
            n.pc_ox(),
            n.pq_ox(),
            n.pq_red(),
            n.pc_red(),
            n.keq(ENZYME),
            n.kcat(ENZYME),
        ],
    )
    return model


def k_b6f(
    pH: float,
    pKreg: float,
    b6f_content: float,
    max_b6f: float,
) -> float:
    pHmod = 1 - (1 / (10 ** (pH - pKreg) + 1))
    b6f_deprot = pHmod * b6f_content
    return b6f_deprot * max_b6f


def vb6f_2024(
    PC: float,
    PCred: float,
    PQ: float,
    PQred: float,
    k_b6f: float,
    Keq_cytb6f: float,
) -> float:
    k_b6f_reverse = k_b6f / Keq_cytb6f
    f_PQH2 = PQred / (
        PQred + PQ
    )  # want to keep the rates in terms of fraction of PQHs, not total number
    f_PQ = 1 - f_PQH2
    return f_PQH2 * PC * k_b6f - f_PQ * PCred * k_b6f_reverse


def add_b6f_2024(
    model: Model,
    *,
    chl_stroma: str = "",
    chl_lumen: str,
) -> Model:
    model.add_parameter(b6f_content := "b6f_content", 1)
    model.add_parameter(max_b6f := "max_b6f", 500)
    model.add_parameter(pKreg := "pKreg", 6.5)

    model.add_derived(
        name=n.keq(ENZYME),
        fn=_keq_cytb6f,
        args=[
            n.ph(chl_lumen),
            "F",
            "E^0_PQ",
            "E^0_PC",
            n.ph(chl_stroma),
            "RT",
            "dG_pH",
        ],
    )

    model.add_derived(
        name=n.keq(ENZYME + "_dyn"),
        fn=k_b6f,
        args=[n.ph(chl_lumen), pKreg, b6f_content, max_b6f],
    )

    model.add_reaction(
        name=ENZYME,
        fn=vb6f_2024,
        stoichiometry={
            n.pc_ox(): -2,
            n.pq_ox(): 1,
            n.h(chl_lumen): Derived(fn=_four_div_by, args=["bH"]),
        },
        args=[
            n.pc_ox(),
            n.pc_red(),
            n.pq_ox(),
            n.pq_red(),
            n.keq(ENZYME + "_dyn"),
            n.keq(ENZYME),
        ],
    )

    return model
