from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s
from mxlbricks.utils import filter_stoichiometry, static

ENZYME = n.ferredoxin_reductase()


def _rate_ferredoxin_reductase(
    Fd: float,
    Fdred: float,
    A1: float,
    A2: float,
    kFdred: float,
    Keq_FAFd: float,
) -> float:
    """rate of the redcution of Fd by the activity of PSI
    used to be equall to the rate of PSI but now
    alternative electron pathway from Fd allows for the production of ROS
    hence this rate has to be separate
    """
    return kFdred * Fd * A1 - kFdred / Keq_FAFd * Fdred * A2


def add_ferredoxin_reductase(
    model: Model,
    *,
    keq: str,  # derived from PSI
    kf: str | None = None,
    e0: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 2.5e5) if kf is None else kf  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kf, e0])

    model.add_reaction(
        name=ENZYME,
        fn=_rate_ferredoxin_reductase,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.fd_ox(): -1,
                n.fd_red(): 1,
            },
        ),
        args=[
            n.fd_ox(),
            n.fd_red(),
            n.a1(),
            n.a2(),
            vmax,
            keq,
        ],
    )
    return model
