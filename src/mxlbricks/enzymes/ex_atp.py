from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s
from mxlbricks.utils import static

ENZYME = n.ex_atp()


def add_atp_consumption(
    model: Model,
    *,
    compartment: str = "",
    kf: str | None = None,
) -> Model:
    kf = static(model, n.kms(ENZYME), 1.0) if kf is None else kf  # FIXME: source

    model.add_reaction(
        name=ENZYME,
        fn=mass_action_1s,
        stoichiometry={
            n.atp(compartment): -1,
        },
        args=[
            n.atp(compartment),
            kf,
        ],
    )
    return model
