"""phosphohexomutase

EC 5.3.1.9

Equilibrator
D-Fructose 6-phosphate(aq) ⇌ D-Glucose 6-phosphate(aq)
Keq = 3 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import rapid_equilibrium_1s_1p
from mxlbricks.utils import static

ENZYME = n.g6pi()


def add_glucose_6_phosphate_isomerase_re(
    model: Model,
    *,
    compartment: str = "",
    kre: str | None = None,
    keq: str | None = None,
) -> Model:
    kre = (
        static(model, n.kre(ENZYME), 800000000.0) if kre is None else kre
    )  # Poolman 2000
    keq = static(model, n.keq(ENZYME), 2.3) if keq is None else keq  # Poolman 2000

    model.add_reaction(
        name=ENZYME,
        fn=rapid_equilibrium_1s_1p,
        stoichiometry={
            n.f6p(compartment): -1,
            n.g6p(compartment): 1,
        },
        args=[
            n.f6p(compartment),
            n.g6p(compartment),
            kre,
            keq,
        ],
    )
    return model
