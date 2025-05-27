"""EC 1.6.5.4
NADH + Proton + 2 Monodehydroascorbate <=> NAD + 2 ascorbate


Equilibrator
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s
from mxlbricks.utils import static

ENZYME = n.mda_reductase2()


def _rate_mda_reductase(
    nadph: float,
    mda: float,
    vmax: float,
    km_nadph: float,
    km_mda: float,
) -> float:
    """Compare Valero et al. 2016"""
    nom = vmax * nadph * mda
    denom = km_nadph * mda + km_mda * nadph + nadph * mda + km_nadph * km_mda
    return nom / denom


def add_mda_reductase2(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    km_mda: str | None = None,
    km_nadph: str | None = None,
) -> Model:
    km_mda = (
        static(model, n.km(ENZYME, n.mda()), 1.4e-3) if km_mda is None else km_mda
    )  # FIXME: source
    km_nadph = (
        static(model, n.km(ENZYME, n.nadph()), 23e-3) if km_nadph is None else km_nadph
    )  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), 1080000 / (60 * 60)) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 2e-3) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_reaction(
        name=ENZYME,
        fn=_rate_mda_reductase,
        stoichiometry={
            n.nadph(): -1,
            n.mda(): -2,
        },
        args=[
            n.nadph(),
            n.mda(),
            vmax,
            km_nadph,
            km_mda,
        ],
    )
    return model
