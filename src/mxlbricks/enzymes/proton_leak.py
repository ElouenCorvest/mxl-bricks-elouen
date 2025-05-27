"""name

EC FIXME

Equilibrator
"""

from mxlpy import Derived, Model

from mxlbricks import names as n
from mxlbricks.fns import protons_stroma
from mxlbricks.utils import static

ENZYME = n.proton_leak()


def _neg_one_div_by(x: float) -> float:
    return -1.0 / x


def _rate_leak(
    protons_lumen: float,
    k_leak: float,
    ph_stroma: float,
) -> float:
    return k_leak * (protons_lumen - protons_stroma(ph_stroma))


def add_proton_leak(
    model: Model,
    *,
    chl_stroma: str = "",
    chl_lumen: str,
    kf: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 10.0) if kf is None else kf

    model.add_reaction(
        name=ENZYME,
        fn=_rate_leak,
        stoichiometry={
            n.h(chl_lumen): Derived(fn=_neg_one_div_by, args=["bH"]),
        },
        args=[
            n.h(chl_lumen),
            kf,
            n.ph(chl_stroma),
        ],
    )
    return model
