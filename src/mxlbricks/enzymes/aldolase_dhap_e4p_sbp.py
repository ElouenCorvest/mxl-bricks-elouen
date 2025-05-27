"""DHAP + EAP <=> SBP

EC 4.1.2.13

Equilibrator
Glycerone phosphate(aq) + D-Erythrose 4-phosphate(aq) ⇌ Sedoheptulose 1,7-bisphosphate(aq)
Keq = 4.8e3 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import rapid_equilibrium_2s_1p
from mxlbricks.utils import static

ENZYME = n.aldolase_dhap_e4p()


def add_aldolase_dhap_e4p_req(
    model: Model,
    *,
    keq: str | None = None,
    compartment: str = "",
    kre: str | None = None,
) -> Model:
    kre = (
        static(model, n.kre(ENZYME), 800000000.0) if keq is None else keq
    )  # Poolman 2000
    keq = static(model, n.keq(ENZYME), 13.0) if keq is None else keq  # Poolman 2000

    model.add_reaction(
        name=ENZYME,
        fn=rapid_equilibrium_2s_1p,
        stoichiometry={
            n.dhap(compartment): -1,
            n.e4p(compartment): -1,
            n.sbp(compartment): 1,
        },
        args=[
            n.dhap(compartment),
            n.e4p(compartment),
            n.sbp(compartment),
            kre,
            keq,
        ],
    )
    return model
