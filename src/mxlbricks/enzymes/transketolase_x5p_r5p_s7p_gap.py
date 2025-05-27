"""GAP + S7P <=> R5P + X5P

EC 2.2.1.1

Equilibrator
D-Glyceraldehyde 3-phosphate(aq) + Sedoheptulose 7-phosphate(aq)
    ⇌ D-Ribose 5-phosphate(aq) + D-Xylulose 5-phosphate(aq)
Keq = 0.2 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import rapid_equilibrium_2s_2p
from mxlbricks.utils import static

ENZYME = n.transketolase_gap_s7p()


def add_transketolase_x5p_r5p_s7p_gap(
    model: Model,
    *,
    chl_stroma: str = "",
    kre: str | None = None,
    keq: str | None = None,
) -> Model:
    kre = static(model, n.kre(ENZYME), 800000000.0) if kre is None else kre
    keq = static(model, n.keq(ENZYME), 0.85) if keq is None else keq

    model.add_reaction(
        name=ENZYME,
        fn=rapid_equilibrium_2s_2p,
        stoichiometry={
            n.gap(chl_stroma): -1,
            n.s7p(chl_stroma): -1,
            n.r5p(chl_stroma): 1,
            n.x5p(chl_stroma): 1,
        },
        args=[
            n.gap(chl_stroma),
            n.s7p(chl_stroma),
            n.r5p(chl_stroma),
            n.x5p(chl_stroma),
            kre,
            keq,
        ],
    )
    return model
