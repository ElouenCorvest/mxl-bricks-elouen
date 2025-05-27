"""Zeaxanthin Epoxidase (stroma):
Zeaxanthin + NADPH + O2 -> Anteraxanthin + NADP + H2O
Antheraxanthin + NADPH + O2 -> Violaxanthin + NADP + H2O
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s
from mxlbricks.utils import static

ENZYME = n.zeaxanthin_epoxidase()


def add_zeaxanthin_epoxidase(
    model: Model,
    kf: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 0.00024) if kf is None else kf

    model.add_reaction(
        name=ENZYME,
        fn=mass_action_1s,
        stoichiometry={
            n.vx(): 1,
        },
        args=[
            n.zx(),
            kf,
        ],
    )
    return model
