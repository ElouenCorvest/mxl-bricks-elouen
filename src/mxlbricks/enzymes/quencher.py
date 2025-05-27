from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s
from mxlbricks.utils import filter_stoichiometry, static

ENZYME = n.quencher()


def add_quenching_reaction(
    model: Model,
    compartment: str = "",
    kf: str | None = None,
) -> Model:
    kf = static(model, kf := n.kre(ENZYME), 1.0) if kf is None else kf

    stoichiometry = filter_stoichiometry(
        model,
        {
            n.energy(compartment): -1.0,
        },
    )

    model.add_reaction(
        name=ENZYME,
        fn=mass_action_1s,
        stoichiometry=stoichiometry,
        args=[
            n.energy(compartment),
            kf,
        ],
    )
    return model
