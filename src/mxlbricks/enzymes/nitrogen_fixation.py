"""name

EC FIXME

Equilibrator
"""

from mxlpy import Derived, Model

from mxlbricks import names as n
from mxlbricks.utils import filter_stoichiometry, static

ENZYME = n.nitrogen_fixation()


def _two_times_convf(convf: float) -> float:
    return 2.0 * convf


def _rate_nitrogen_fixation(
    oxo: float,
    atp: float,
    fd_red: float,
    nh4: float,
    k_fwd: float,
    convf: float,
) -> float:
    return k_fwd * oxo * atp * nh4 * (2 * fd_red * convf)


def add_nitrogen_metabolism(
    model: Model,
    kf: str | None = None,
    convf: str | None = None,
) -> Model:
    """Equilibrator
        2-Oxoglutarate(aq) + ATP(aq) + 2 ferredoxin(red)(aq) + NH4 (aq)
        ⇌ Glutamate(aq) + ADP(aq) + 2 ferredoxin(ox)(aq) + Orthophosphate(aq)
    K'eq = 2.4e13

    Units
     - 2-oxoglutarate: mM
     - ATP: mM
     - Fd_red: ?
     - NH4: mM
     - Glutamate: mM
     - ADP: mM
     - Fd_ox: ?
     - Orthophosphate: mM
    """
    kf = static(model, n.kf(ENZYME), 1.0) if kf is None else kf
    convf = static(model, n.convf(), 3.2e-2) if convf is None else convf

    model.add_reaction(
        ENZYME,
        _rate_nitrogen_fixation,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.atp(): -1.0,  # mM
                n.nh4(): -1.0,  # mM
                n.glutamate(): 1.0,  # mM
                n.fd_ox(): Derived(fn=_two_times_convf, args=[convf]),
            },
        ),
        args=[
            n.oxoglutarate(),
            n.atp(),
            n.fd_red(),
            n.nh4(),
            kf,
            convf,
        ],
    )

    return model
