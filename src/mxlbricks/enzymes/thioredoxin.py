"""name

EC FIXME

Equilibrator
"""

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import (
    mass_action_1s,
    mass_action_2s,
    michaelis_menten_1s,
)
from mxlbricks.utils import filter_stoichiometry, static


def add_cbb_pfd_speedup(
    model: Model,
    km: str | None = None,
    vmax: str | None = None,
) -> Model:
    km = static(model, n.km(n.light_speedup()), 150.0)
    vmax = static(model, n.vmax(n.light_speedup()), 6.0)
    model.add_derived(
        n.light_speedup(),
        michaelis_menten_1s,
        args=[
            n.pfd(),
            vmax,
            km,
        ],
    )
    return model


def add_fd_tr_reductase_2021(
    model: Model,
    kf: str | None = None,
) -> Model:
    """Equilibrator
    Thioredoxin(ox)(aq) + 2 ferredoxin(red)(aq) ⇌ Thioredoxin(red)(aq) + 2 ferredoxin(ox)(aq)
    Keq = 4.9e3 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
    """
    enzyme_name = n.ferredoxin_thioredoxin_reductase()
    kf = static(model, n.kf(enzyme_name), 1)

    model.add_reaction(
        name=enzyme_name,
        fn=mass_action_2s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.tr_ox(): -1,
                n.fd_red(): -1,
                n.tr_red(): 1,
                n.fd_ox(): 1,
            },
        ),
        args=[
            n.tr_ox(),
            n.fd_red(),
            kf,
        ],
    )
    return model


def add_fd_tr_reductase(
    model: Model,
    kf: str | None = None,
) -> Model:
    """Equilibrator
    Thioredoxin(ox)(aq) + 2 ferredoxin(red)(aq) ⇌ Thioredoxin(red)(aq) + 2 ferredoxin(ox)(aq)
    Keq = 4.9e3 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
    """
    enzyme_name = n.ferredoxin_thioredoxin_reductase()
    kf = static(model, n.kf(enzyme_name), 1)

    model.add_reaction(
        name=enzyme_name,
        fn=mass_action_2s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.tr_ox(): -1,
                n.fd_red(): -2,
                n.tr_red(): 1,
                n.fd_ox(): 2,
            },
        ),
        args=[
            n.tr_ox(),
            n.fd_red(),
            kf,
        ],
    )
    return model


def add_nadph_tr_reductase(
    model: Model,
    kf: str | None = None,
) -> Model:
    """Equilibrator
    Thioredoxin(ox)(aq) + NADPH(aq) ⇌ Thioredoxin(red)(aq) + NADP(aq)
    Keq = 2e1 (@ pH = 7.5, pMg = 3.0, Ionic strength = 0.25)
    """
    enzyme_name = n.nadph_thioredoxin_reductase()
    kf = static(model, n.kf(enzyme_name), 1)

    model.add_reaction(
        name=enzyme_name,
        fn=mass_action_2s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.tr_ox(): -1,
                n.nadph(): -1,
                n.tr_red(): 1,
                n.nadp(): 1,
            },
        ),
        args=[
            n.tr_ox(),
            n.nadph(),
            kf,
        ],
    )
    return model


def add_tr_e_activation(
    model: Model,
    kf: str | None = None,
) -> Model:
    enzyme_name = n.tr_activation()
    kf = static(model, n.kf(enzyme_name), 1)
    model.add_reaction(
        name=enzyme_name,
        fn=mass_action_2s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.e_inactive(): -1,
                n.tr_red(): -1,
                n.e_active(): 1,
                n.tr_ox(): 1,
            },
        ),
        args=[
            n.e_inactive(),
            n.tr_red(),
            kf,
        ],
    )
    return model


def add_tr_e_activation2021(
    model: Model,
    kf: str | None = None,
) -> Model:
    enzyme_name = n.tr_activation()
    kf = static(model, n.kf(enzyme_name), 1)
    model.add_reaction(
        name=enzyme_name,
        fn=mass_action_2s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.e_inactive(): -5,
                n.tr_red(): -5,
                n.e_active(): 5,
                n.tr_ox(): 5,
            },
        ),
        args=[
            n.e_inactive(),
            n.tr_red(),
            kf,
        ],
    )
    return model


def add_e_relaxation(
    model: Model,
    kf: str | None = None,
) -> Model:
    enzyme_name = n.tr_inactivation()
    kf = static(model, n.kf(enzyme_name), 0.1)

    model.add_reaction(
        name=enzyme_name,
        fn=mass_action_1s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.e_active(): -1,
                n.e_inactive(): 1,
            },
        ),
        args=[
            n.e_active(),
            kf,
        ],
    )
    return model


def add_e_relaxation_2021(
    model: Model,
    kf: str | None = None,
) -> Model:
    enzyme_name = n.tr_inactivation()
    kf = static(model, n.kf(enzyme_name), 0.1)

    model.add_reaction(
        name=enzyme_name,
        fn=mass_action_1s,
        stoichiometry=filter_stoichiometry(
            model,
            {
                n.e_active(): -5,
                n.e_inactive(): 5,
            },
        ),
        args=[
            n.e_active(),
            kf,
        ],
    )
    return model


def add_thioredoxin_regulation(model: Model) -> Model:
    add_fd_tr_reductase(model)
    add_tr_e_activation(model)
    add_e_relaxation(model)
    return model


def add_thioredoxin_regulation2021(model: Model) -> Model:
    add_fd_tr_reductase_2021(model)
    add_tr_e_activation2021(model)
    add_e_relaxation_2021(model)
    return model


def add_thioredoxin_regulation_from_nadph(model: Model) -> Model:
    add_nadph_tr_reductase(model)
    add_tr_e_activation(model)
    add_e_relaxation(model)
    return model
