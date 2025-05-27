"""name

EC 4.2.1.1

Equilibrator

hco3:co2 is ~50:1 according to Straßburger
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import reversible_mass_action_keq_1s_1p
from mxlbricks.utils import static

ENZYME = n.carbonic_anhydrase()


def add_carbonic_anhydrase_mass_action(
    model: Model,
    compartment: str = "",
    kf: str | None = None,
    keq: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), 1000) if kf is None else kf  # FIXME: source
    keq = static(model, n.keq(ENZYME), 50) if keq is None else keq  # FIXME: source

    model.add_reaction(
        name=ENZYME,
        fn=reversible_mass_action_keq_1s_1p,
        stoichiometry={
            n.co2(compartment): -1,
            n.hco3(compartment): 1,
        },
        args=[
            n.co2(compartment),
            n.hco3(compartment),
            kf,
            keq,
        ],
    )
    return model
