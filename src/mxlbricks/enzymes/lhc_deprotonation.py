from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s
from mxlbricks.utils import filter_stoichiometry, static

ENZYME = n.lhc_deprotonation()


def add_lhc_deprotonation(
    model: Model,
    kf: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 0.0096) if kf is None else kf

    model.add_reaction(
        name=ENZYME,
        fn=mass_action_1s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.psbs_pr(): -1,
                n.psbs_de(): 1,
            },
        ),
        args=[
            n.psbs_pr(),
            kf,
        ],
    )
    return model
