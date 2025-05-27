"""ribulose-phosphate 3-epimerase

EC 5.1.3.1

Equilibrator
    D-Xylulose 5-phosphate(aq) ⇌ D-Ribulose 5-phosphate(aq)
    Keq = 0.3 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import rapid_equilibrium_1s_1p
from mxlbricks.utils import static

ENZYME = n.ribulose_phosphate_epimerase()


def add_ribulose_5_phosphate_3_epimerase(
    model: Model,
    *,
    chl_stroma: str = "",
    kre: str | None = None,
    keq: str | None = None,
) -> Model:
    kre = static(model, n.kre(ENZYME), 800000000.0) if kre is None else kre
    keq = static(model, n.keq(ENZYME), 0.67) if keq is None else keq

    model.add_reaction(
        name=ENZYME,
        fn=rapid_equilibrium_1s_1p,
        stoichiometry={
            n.x5p(chl_stroma): -1,
            n.ru5p(chl_stroma): 1,
        },
        args=[
            n.x5p(chl_stroma),
            n.ru5p(chl_stroma),
            kre,
            keq,
        ],
    )
    return model
