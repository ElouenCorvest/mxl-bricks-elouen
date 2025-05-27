"""ribose-5-phosphate isomerase

EC 5.3.1.6

Equilibrator
    D-Ribose 5-phosphate(aq) ⇌ D-Ribulose 5-phosphate(aq)
    Keq = 0.4 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import rapid_equilibrium_1s_1p
from mxlbricks.utils import static

ENZYME = n.ribose_phosphate_isomerase()


def add_ribose_5_phosphate_isomerase(
    model: Model,
    *,
    chl_stroma: str = "",
    kre: str | None = None,
    keq: str | None = None,
) -> Model:
    kre = static(model, n.kre(ENZYME), 800000000.0) if kre is None else kre
    keq = static(model, n.keq(ENZYME), 0.4) if keq is None else keq

    model.add_reaction(
        name=ENZYME,
        fn=rapid_equilibrium_1s_1p,
        stoichiometry={
            n.r5p(chl_stroma): -1,
            n.ru5p(chl_stroma): 1,
        },
        args=[
            n.r5p(chl_stroma),
            n.ru5p(chl_stroma),
            kre,
            keq,
        ],
    )
    return model
