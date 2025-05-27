"""dehydroascorbate_reductase, DHAR

EC FIXME

Equilibrator
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s
from mxlbricks.utils import static

ENZYME = n.dehydroascorbate_reductase()


def _rate_dhar(
    dha: float,
    gsh: float,
    vmax: float,
    km_dha: float,
    km_gsh: float,
    k: float,
) -> float:
    nom = vmax * dha * gsh
    denom = k + km_dha * gsh + km_gsh * dha + dha * gsh
    return nom / denom


def add_dehydroascorbate_reductase(
    model: Model,
    *,
    kcat: str | None = None,
    e0: str | None = None,
    km_dha: str | None = None,
    km_gsh: str | None = None,
) -> Model:
    km_dha = (
        static(model, n.km(ENZYME, n.dha()), 70e-3) if km_dha is None else km_dha
    )  # FIXME: source
    km_gsh = (
        static(model, n.km(ENZYME, n.glutathion_red()), 2.5e3 * 1e-3)
        if km_gsh is None
        else km_gsh
    )  # FIXME: source
    kcat = static(model, n.kcat(ENZYME), 142) if kcat is None else kcat  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.7e-3) if e0 is None else e0  # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])

    model.add_parameter("K", 5e5 * (1e-3) ** 2)

    model.add_reaction(
        name=ENZYME,
        fn=_rate_dhar,
        stoichiometry={
            n.dha(): -1,
            n.glutathion_ox(): 1,
        },
        args=[
            n.dha(),
            n.glutathion_red(),
            vmax,
            km_dha,
            km_gsh,
            "K",
        ],
    )
    return model
