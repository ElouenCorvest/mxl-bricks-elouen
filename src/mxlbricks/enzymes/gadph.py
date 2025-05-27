"""Glyceraldehyde 3-phosphate dehydrogenase (GADPH)

EC 1.2.1.13

Equilibrator
    NADPH(aq) + 3-Phospho-D-glyceroyl phosphate(aq)
    ⇌ NADP (aq) + Orthophosphate(aq) + D-Glyceraldehyde 3-phosphate(aq)
    Keq = 2 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)

FIXME: Poolman uses H+ in the description. Why?
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import rapid_equilibrium_3s_3p
from mxlbricks.utils import filter_stoichiometry, static

ENZYME = n.gadph()


def add_gadph(
    model: Model,
    *,
    compartment: str = "",
    kre: str | None = None,
    keq: str | None = None,
) -> Model:
    kre = (
        static(model, n.kre(ENZYME), 800000000.0) if kre is None else kre
    )  # Poolman 2000
    keq = (
        static(model, n.keq(ENZYME), 16000000.0) if keq is None else keq
    )  # Poolman 2000

    model.add_reaction(
        name=ENZYME,
        fn=rapid_equilibrium_3s_3p,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.nadph(): -1.0,
                n.bpga(compartment): -1.0,
                n.nadp(): 1.0,
                n.pi(): 1.0,
                n.gap(compartment): 1.0,
            },
        ),
        args=[
            n.bpga(compartment),
            n.nadph(compartment),
            n.h(compartment),
            n.gap(compartment),
            n.nadp(compartment),
            n.pi(compartment),
            kre,
            keq,
        ],
    )
    return model
