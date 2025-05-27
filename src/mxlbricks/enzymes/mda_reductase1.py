"""name

EC FIXME

Equilibrator
Monodehydroascorbate(aq) + 0.5 NAD (aq) ⇌ Dehydroascorbate(aq) + 0.5 NADH(aq)
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.utils import static

ENZYME = n.mda_reductase1()


def _rate_mda_reductase(
    mda: float,
    k3: float,
) -> float:
    return k3 * mda**2


def add_mda_reductase1(
    model: Model,
    kf: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 0.5 / 1e-3)  # FIXME: source

    model.add_reaction(
        name=ENZYME,
        fn=_rate_mda_reductase,
        stoichiometry={
            n.mda(): -2,
            n.dha(): 1,
        },
        args=[
            n.mda(),
            kf,
        ],
    )
    return model
