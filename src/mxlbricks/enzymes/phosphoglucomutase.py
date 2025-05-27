"""glucose phosphomutase

EC 5.4.2.2

G6P <=> G1P

Equilibrator
Glucose 6-phosphate(aq) ⇌ D-Glucose-1-phosphate(aq)
Keq = 0.05 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import rapid_equilibrium_1s_1p
from mxlbricks.utils import static

ENZYME = n.phosphoglucomutase()


def add_phosphoglucomutase(
    model: Model,
    *,
    compartment: str = "",
    kre: str | None = None,
    keq: str | None = None,
) -> Model:
    kre = static(model, n.kre(ENZYME), 800000000.0) if kre is None else kre
    keq = static(model, n.keq(ENZYME), 0.058) if keq is None else keq

    model.add_reaction(
        name=ENZYME,
        fn=rapid_equilibrium_1s_1p,
        stoichiometry={
            n.g6p(compartment): -1,
            n.g1p(compartment): 1,
        },
        args=[
            n.g6p(compartment),
            n.g1p(compartment),
            kre,
            keq,
        ],
    )
    return model
