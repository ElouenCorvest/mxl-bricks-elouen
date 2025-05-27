from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.utils import static

ENZYME = n.cyclic_electron_flow()


def _rate_cyclic_electron_flow(
    Pox: float,
    Fdred: float,
    kcyc: float,
) -> float:
    return kcyc * Fdred**2 * Pox


def add_cyclic_electron_flow(
    model: Model,
    kf: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 1.0) if kf is None else kf  # FIXME: source

    model.add_reaction(
        name=ENZYME,
        fn=_rate_cyclic_electron_flow,
        stoichiometry={
            n.pq_ox(): -1,
            n.fd_ox(): 2,
        },
        args=[
            n.pq_ox(),
            n.fd_red(),
            kf,
        ],
    )
    return model
