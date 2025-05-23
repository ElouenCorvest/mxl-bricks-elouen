"""name

EC FIXME

Equilibrator
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import protons_stroma
from mxlbricks.utils import filter_stoichiometry, static

ENZYME = n.lhc_protonation()


def _protonation_hill(
    vx: float,
    h: float,
    nh: float,
    k_fwd: float,
    k_ph_sat: float,
) -> float:
    return k_fwd * (h**nh / (h**nh + protons_stroma(k_ph_sat) ** nh)) * vx  # type: ignore


def add_lhc_protonation(
    model: Model,
    *,
    chl_lumen: str,
    kf: str | None = None,
    kh_lhc: str | None = None,
    k_ph_sat: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 0.0096) if kf is None else kf
    kh_lhc = static(model, n.kh(ENZYME), 3.0) if kh_lhc is None else kh_lhc
    k_ph_sat = static(model, n.ksat(ENZYME), 5.8) if k_ph_sat is None else k_ph_sat

    model.add_reaction(
        name=ENZYME,
        fn=_protonation_hill,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.psbs_de(): -1,
                n.psbs_pr(): 1,
            },
        ),
        args=[
            n.psbs_de(),
            n.h(chl_lumen),
            kh_lhc,
            kf,
            k_ph_sat,
        ],
    )
    return model
