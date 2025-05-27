"""name

EC FIXME

Equilibrator
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s
from mxlbricks.utils import static


def _rate_state_transition_ps1_ps2(
    ant: float,
    pox: float,
    p_tot: float,
    k_stt7: float,
    km_st: float,
    n_st: float,
) -> float:
    return k_stt7 * (1 / (1 + (pox / p_tot / km_st) ** n_st)) * ant


def add_state_transitions(
    model: Model,
    kstt7: str | None = None,
    kms: str | None = None,
    n_st: str | None = None,
    kpph: str | None = None,
) -> Model:
    kstt7 = static(model, "kStt7", 0.0035) if kstt7 is None else kstt7
    kms = static(model, "KM_ST", 0.2) if kms is None else kms
    n_st = static(model, "n_ST", 2.0) if n_st is None else n_st
    kpph = static(model, "kPph1", 0.0013) if kpph is None else kpph

    enzyme_name = n.lhc_state_transition_12()
    model.add_reaction(
        name=enzyme_name,
        fn=_rate_state_transition_ps1_ps2,
        stoichiometry={
            n.lhc(): -1,
        },
        args=[
            n.lhc(),
            n.pq_ox(),
            n.total_pq(),
            kstt7,
            kms,
            n_st,
        ],
    )

    enzyme_name = n.lhc_state_transition_21()
    model.add_reaction(
        name=enzyme_name,
        fn=mass_action_1s,
        stoichiometry={
            n.lhc(): 1,
        },
        args=[
            n.lhcp(),
            kpph,
        ],
    )
    return model
