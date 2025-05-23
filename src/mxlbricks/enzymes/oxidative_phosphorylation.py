from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import reversible_mass_action_keq_2s_2p
from mxlbricks.utils import filter_stoichiometry, static

ENZYME = n.oxidative_phosphorylation()


def add_oxidative_phosphorylation(
    model: Model,
    kf: str | None = None,
    keq: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 1) if kf is None else kf
    keq = static(model, n.keq(ENZYME), 3 / 2) if keq is None else keq

    model.add_reaction(
        ENZYME,
        reversible_mass_action_keq_2s_2p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.nadph(): -1,
                n.adp(): -1,
                n.nadp(): 1,
                n.atp(): 1,
            },
        ),
        args=[
            n.nadph(),
            n.adp(),
            n.nadp(),
            n.atp(),
            kf,
            keq,
        ],
    )
    return model
