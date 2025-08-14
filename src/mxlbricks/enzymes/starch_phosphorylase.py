from mxlpy import Model, units

from mxlbricks import names as n
from mxlbricks.fns import michaelis_menten_1s_1i
from mxlbricks.utils import (
    default_kis,
    default_km,
    default_name,
    default_vmax,
    filter_stoichiometry,
)


def add_starch_phosphorylase(
    model: Model,
    *,
    rxn: str | None = None,
    # Vars
    starch: str | None = None,
    pi: str | None = None,
    g1p: str | None = None,
    # Params
    e0: str | None = None,
    kcat: str | None = None,
    km_pi: str | None = None,
    ki_g1p: str | None = None
) -> Model:
    rxn = default_name(rxn, n.starch_phosphorylase)
    # Vars
    starch = default_name(starch, n.starch)
    pi = default_name(pi, n.pi)
    g1p = default_name(g1p, n.g1p)
    
    model.add_reaction(
        rxn,
        fn=michaelis_menten_1s_1i,
        stoichiometry=filter_stoichiometry(
            model,
            {
                pi: -1,
                g1p: 1
            },
            optional={starch: -1}
        ),
        args=[
            pi,
            g1p,
            default_vmax(
                model,
                e0=e0,
                kcat=kcat,
                rxn=rxn,
                e0_value=1,
                kcat_value=0.04,
            ),
            default_km(model, par=km_pi, rxn=rxn, subs=pi, value=0.1, unit=units.mmol / units.liter, source="Poolman1999"),
            default_kis(model, par=ki_g1p, rxn=rxn, substrate=g1p, value=0.05, unit=units.mmol / units.liter, source="Poolman1999"),
        ]
    )
    
    return model