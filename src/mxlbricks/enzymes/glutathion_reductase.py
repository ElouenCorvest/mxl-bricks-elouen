"""name

EC 1.8.1.7

glutathione + NADP <=> glutathion-disulfide + NADPH + H+

Equilibrator
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s
from mxlbricks.utils import static

ENZYME = n.glutathion_reductase()


def _rate_glutathion_reductase(
    nadph: float,
    gssg: float,
    vmax: float,
    km_nadph: float,
    km_gssg: float,
) -> float:
    nom = vmax * nadph * gssg
    denom = km_nadph * gssg + km_gssg * nadph + nadph * gssg + km_nadph * km_gssg
    return nom / denom


def add_glutathion_reductase_irrev(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    km_gssg: str | None = None,
    km_nadph: str | None = None,
) -> Model:
    km_gssg = (
        static(model, n.km(ENZYME, n.glutathion_ox()), 2e2 * 1e-3)
        if km_gssg is None
        else km_gssg
    )  # FIXME: source
    km_nadph = (
        static(model, n.km(ENZYME, n.nadph()), 3e-3) if km_nadph is None else km_nadph
    )  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 595) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.4e-3) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=_rate_glutathion_reductase,
        stoichiometry={n.nadph(): -1, n.glutathion_ox(): -1},
        args=[
            n.nadph(),
            n.glutathion_ox(),
            vmax,
            km_nadph,
            km_gssg,
        ],
    )
    return model
