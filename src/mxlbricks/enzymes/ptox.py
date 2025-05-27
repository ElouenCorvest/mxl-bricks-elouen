"""Plastid terminal oxidase

2 QH2 + O2 -> 2 Q + 2 H2O
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import mass_action_2s
from mxlbricks.utils import static

ENZYME = n.ptox()


def add_ptox(
    model: Model,
    *,
    compartment: str = "",
    kf: str | None = None,
) -> Model:
    kf = static(model, "kPTOX", 0.01) if kf is None else kf

    model.add_reaction(
        name=ENZYME,
        fn=mass_action_2s,
        stoichiometry={n.pq_ox(): 1},
        args=[
            n.pq_red(),
            n.o2(compartment),
            kf,
        ],
    )
    return model
