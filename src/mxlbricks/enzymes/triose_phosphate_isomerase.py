"""triose-phosphate isomerase

EC 5.3.1.1

Equilibrator
    D-Glyceraldehyde 3-phosphate(aq) ⇌ Glycerone phosphate(aq)
    Keq = 10 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import rapid_equilibrium_1s_1p
from mxlbricks.utils import static

ENZYME = n.triose_phosphate_isomerase()


def add_triose_phosphate_isomerase(
    model: Model,
    *,
    chl_stroma: str = "",
    kre: str | None = None,
    keq: str | None = None,
) -> Model:
    kre = static(model, n.kre(ENZYME), 800000000.0) if kre is None else kre
    keq = static(model, n.keq(ENZYME), 22.0) if keq is None else keq

    model.add_reaction(
        name=ENZYME,
        fn=rapid_equilibrium_1s_1p,
        stoichiometry={
            n.gap(chl_stroma): -1,
            n.dhap(chl_stroma): 1,
        },
        args=[
            n.gap(chl_stroma),
            n.dhap(chl_stroma),
            kre,
            keq,
        ],
    )
    return model
