"""Violaxanthin Deepoxidase (lumen)
Violaxanthin + Ascorbate -> Antheraxanthin + Dehydroascorbate + H2O
Antheraxanthin + Ascorbate -> Zeaxanthin + Dehydroascorbate + H2O
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import protons_stroma
from mxlbricks.utils import static

ENZYME = n.violaxanthin_deepoxidase()


def _rate_protonation_hill(
    Vx: float,
    H: float,
    k_fwd: float,
    nH: float,
    kphSat: float,
) -> float:
    return k_fwd * (H**nH / (H**nH + protons_stroma(kphSat) ** nH)) * Vx  # type: ignore


def add_violaxanthin_epoxidase(
    model: Model,
    *,
    chl_lumen: str,
    kf: str | None = None,
    kh_zx: str | None = None,
    kphsat: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 0.0024)
    kh_zx = static(model, n.kh(ENZYME), 5.0)
    kphsat = static(model, n.ksat(ENZYME), 5.8)

    model.add_reaction(
        name=ENZYME,
        fn=_rate_protonation_hill,
        stoichiometry={
            n.vx(): -1,
        },
        args=[
            n.vx(),
            n.h(chl_lumen),
            kf,
            kh_zx,
            kphsat,
        ],
    )
    return model
