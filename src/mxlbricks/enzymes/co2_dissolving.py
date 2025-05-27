from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import diffusion
from mxlbricks.utils import static

ENZYME = n.co2_dissolving()


def add_co2_dissolving(
    model: Model,
    *,
    compartment: str = "",
    kf: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 4.5) if kf is None else kf  # FIXME: source

    model.add_reaction(
        name=ENZYME,
        fn=diffusion,
        stoichiometry={
            n.co2(compartment): 1,
        },
        args=[
            n.co2(compartment),
            n.co2_atmosphere(),
            kf,
        ],
    )
    return model
