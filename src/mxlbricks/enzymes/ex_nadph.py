"""name

Equilibrator
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s
from mxlbricks.utils import static

ENZYME = n.ex_nadph()


def add_nadph_consumption(
    model: Model,
    *,
    compartment: str = "",
    kf: str,
) -> Model:
    kf = static(model, n.kf(ENZYME), 1.0) if kf is None else kf  # FIXME: source

    model.add_reaction(
        name=ENZYME,
        fn=mass_action_1s,
        stoichiometry={
            n.nadph(compartment): -1,
        },
        args=[
            n.nadph(compartment),
            kf,
        ],
    )
    return model
