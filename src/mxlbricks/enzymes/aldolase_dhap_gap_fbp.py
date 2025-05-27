"""DHAP + GAP <=> FBP

EC 4.1.2.13

Equilibrator
Glycerone phosphate(aq) + D-Glyceraldehyde 3-phosphate(aq) ⇌ D-Fructose 1,6-bisphosphate(aq)
Keq = 1.1e4 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)

"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import rapid_equilibrium_2s_1p
from mxlbricks.utils import static

ENZYME = n.aldolase_dhap_gap()


def add_aldolase_dhap_gap_req(
    model: Model,
    *,
    compartment: str = "",
    kre: str | None = None,
    keq: str | None = None,
) -> Model:
    kre = (
        static(model, n.kre(ENZYME), 800000000.0) if keq is None else keq
    )  # Poolman 2000
    keq = static(model, n.keq(ENZYME), 7.1) if keq is None else keq  # Poolman 2000

    model.add_reaction(
        name=ENZYME,
        fn=rapid_equilibrium_2s_1p,
        stoichiometry={
            n.gap(compartment): -1,
            n.dhap(compartment): -1,
            n.fbp(compartment): 1,
        },
        args=[
            n.gap(compartment),
            n.dhap(compartment),
            n.fbp(compartment),
            kre,
            keq,
        ],
    )
    return model
