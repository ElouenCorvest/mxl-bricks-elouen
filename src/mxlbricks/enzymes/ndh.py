"""NAD(P)H dehydrogenase-like complex (NDH)

PQH2 -> PQ

"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s
from mxlbricks.utils import static

ENZYME = n.ndh()


def add_ndh(
    model: Model,
    kf: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 0.002) if kf is None else kf

    model.add_reaction(
        name=ENZYME,
        fn=mass_action_1s,
        stoichiometry={
            n.pq_ox(): -1,
        },
        args=[
            n.pq_ox(),
            kf,
        ],
    )
    return model
