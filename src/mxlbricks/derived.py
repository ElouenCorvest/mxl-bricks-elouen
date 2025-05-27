import math

from mxlpy import Model

from mxlbricks import names as n
from mxlbricks.fns import div, mass_action_1s, moiety_1, moiety_2
from mxlbricks.utils import static


def add_ascorbate_moiety(
    model: Model,
    *,
    total: str | None = None,
) -> Model:
    total = static(model, n.total_ascorbate(), 10) if total is None else total

    model.add_derived(
        name=n.ascorbate(),
        fn=moiety_2,
        args=[n.mda(), n.dha(), total],
    )
    return model


def add_adenosin_moiety(
    model: Model,
    *,
    compartment: str = "",
    total: str | None = None,
) -> Model:
    total = static(model, n.total_ascorbate(), 0.5) if total is None else total

    model.add_derived(
        name=n.adp(compartment),
        fn=moiety_1,
        args=[
            n.atp(compartment),
            n.total_adenosines(),
        ],
    )
    return model


def add_enzyme_moiety(
    model: Model,
    *,
    total: str | None = None,
) -> Model:
    total = static(model, n.e_total(), 6) if total is None else total

    model.add_derived(
        name=n.e_active(),
        fn=moiety_1,
        args=[
            n.e_inactive(),
            total,
        ],
    )
    return model


def add_ferredoxin_moiety(
    model: Model,
    *,
    total: str | None = None,
) -> Model:
    total = static(model, n.total_ferredoxin(), 5.0) if total is None else total

    model.add_derived(
        name=n.fd_red(),
        fn=moiety_1,
        args=[
            n.fd_ox(),
            total,
        ],
    )
    return model


def add_glutamate_moiety(
    model: Model,
    *,
    chl_stroma: str = "",
    total: str | None = None,
) -> Model:
    total = static(model, n.total_glutamate(), 3.0) if total is None else total

    model.add_derived(
        name=n.oxoglutarate(chl_stroma),
        fn=moiety_1,
        args=[
            n.glutamate(),
            total,
        ],
    )
    return model


def _glutathion_moiety(
    gssg: float,
    gs_total: float,
) -> float:
    return gs_total - 2 * gssg


def add_glutathion_moiety(
    model: Model,
    *,
    total: str | None = None,
) -> Model:
    total = static(model, n.total_glutathion(), 10.0) if total is None else total

    model.add_derived(
        name=n.glutathion_red(),
        fn=_glutathion_moiety,
        args=[
            n.glutathion_ox(),
            total,
        ],
    )
    return model


def add_hco3_from_co2(
    model: Model,
    *,
    compartment: str = "",
    factor: str | None = None,
) -> Model:
    factor = static(model, "CO2/HCO3 ratio", 50)

    return model.add_derived(
        n.hco3(compartment),
        fn=mass_action_1s,
        args=[
            n.co2(),
            factor,
        ],
    )


def add_lhc_moiety(
    model: Model,
    *,
    total: str | None = None,
) -> Model:
    total = static(model, n.total_lhc(), 1.0) if total is None else total

    model.add_derived(
        name=n.lhcp(),
        fn=moiety_1,
        args=[
            n.lhc(),
            total,
        ],
    )
    return model


def add_nad_moiety(
    model: Model,
    *,
    total: str | None = None,
) -> Model:
    total = static(model, n.total_nad(), 0.86) if total is None else total

    model.add_derived(
        name=n.nad(),
        fn=moiety_1,
        args=[
            n.nadh(),
            total,
        ],
    )
    return model


def add_nadp_moiety(
    model: Model,
    *,
    total: str | None = None,
) -> Model:
    total = static(model, n.total_nadp(), 0.5) if total is None else total

    model.add_derived(
        name=n.nadp(),
        fn=moiety_1,
        args=[n.nadph(), total],
    )
    return model


def add_plastocyanin_moiety(
    model: Model,
    *,
    total: str | None = None,
) -> Model:
    total = static(model, n.total_pc(), 4.0) if total is None else total

    model.add_derived(
        name=n.pc_red(),
        fn=moiety_1,
        args=[n.pc_ox(), total],
    )
    return model


def add_plastoquinone_moiety(
    model: Model,
    *,
    total: str | None = None,
) -> Model:
    total = static(model, n.total_pq(), 17.5) if total is None else total

    model.add_derived(
        name=n.pq_red(),
        fn=moiety_1,
        args=[n.pq_ox(), total],
    )
    return model


def add_carotenoid_moiety(
    model: Model,
    *,
    compartment: str = "",
    total: str | None = None,
) -> Model:
    total = static(model, n.total_carotenoids(), 1.0) if total is None else total

    model.add_derived(
        name=n.zx(compartment),
        fn=moiety_1,
        args=[n.vx(compartment), total],
    )
    return model


def add_thioredoxin_moiety(
    model: Model,
    *,
    compartment: str = "",
    total: str | None = None,
) -> Model:
    total = (
        static(model, n.total_thioredoxin(compartment), 1.0) if total is None else total
    )

    model.add_derived(
        name=n.tr_red(compartment),
        fn=moiety_1,
        args=[n.tr_ox(compartment), total],
    )
    return model


def add_psbs_moietry(
    model: Model,
    *,
    total: str | None = None,
) -> Model:
    """Derive protonated form from deprotonated form"""
    total = static(model, n.total_psbs(), 1.0) if total is None else total

    model.add_derived(
        name=n.psbs_pr(),
        fn=moiety_1,
        args=[
            n.psbs_de(),
            total,
        ],
    )
    return model


def add_rt(
    model: Model,
    r: str | None = None,
    t: str | None = None,
) -> Model:
    r = static(model, "R", 0.0083) if r is None else r
    t = static(model, "T", 298.0) if t is None else t

    model.add_derived(
        "RT",
        mass_action_1s,
        args=[r, t],
    )
    return model


def _keq_pq_red(
    E0_QA: float,
    F: float,
    E0_PQ: float,
    pHstroma: float,
    dG_pH: float,
    RT: float,
) -> float:
    dg1 = -E0_QA * F
    dg2 = -2 * E0_PQ * F
    dg = -2 * dg1 + dg2 + 2 * pHstroma * dG_pH

    return math.exp(-dg / RT)


def add_plastoquinone_keq(
    model: Model,
    *,
    chl_stroma: str = "",
) -> Model:
    model.add_parameter("E^0_QA", -0.14)
    model.add_parameter("E^0_PQ", 0.354)

    model.add_derived(
        n.keq(n.pq_red()),
        _keq_pq_red,
        args=[
            "E^0_QA",
            "F",
            "E^0_PQ",
            n.ph(chl_stroma),
            "dG_pH",
            "RT",
        ],
    )
    return model


def _pi_cbb(
    phosphate_total: float,
    pga: float,
    bpga: float,
    gap: float,
    dhap: float,
    fbp: float,
    f6p: float,
    g6p: float,
    g1p: float,
    sbp: float,
    s7p: float,
    e4p: float,
    x5p: float,
    r5p: float,
    rubp: float,
    ru5p: float,
    atp: float,
) -> float:
    return phosphate_total - (
        pga
        + 2 * bpga
        + gap
        + dhap
        + 2 * fbp
        + f6p
        + g6p
        + g1p
        + 2 * sbp
        + s7p
        + e4p
        + x5p
        + r5p
        + 2 * rubp
        + ru5p
        + atp
    )


def _pi_cbb_pr(
    phosphate_total: float,
    pga: float,
    bpga: float,
    gap: float,
    dhap: float,
    fbp: float,
    f6p: float,
    g6p: float,
    g1p: float,
    sbp: float,
    s7p: float,
    e4p: float,
    x5p: float,
    r5p: float,
    rubp: float,
    ru5p: float,
    atp: float,
    pgo: float,
) -> float:
    return phosphate_total - (
        pga
        + 2 * bpga
        + gap
        + dhap
        + 2 * fbp
        + f6p
        + g6p
        + g1p
        + 2 * sbp
        + s7p
        + e4p
        + x5p
        + r5p
        + 2 * rubp
        + ru5p
        + atp
        + pgo
    )


def add_orthophosphate_moiety_cbb(
    model: Model,
    *,
    chl_stroma: str = "",
    total: str | None = None,
) -> Model:
    total = static(model, n.total_orthophosphate(), 15.0) if total is None else total

    args = [
        total,
        n.pga(chl_stroma),
        n.bpga(chl_stroma),
        n.gap(chl_stroma),
        n.dhap(chl_stroma),
        n.fbp(chl_stroma),
        n.f6p(chl_stroma),
        n.g6p(chl_stroma),
        n.g1p(chl_stroma),
        n.sbp(chl_stroma),
        n.s7p(chl_stroma),
        n.e4p(chl_stroma),
        n.x5p(chl_stroma),
        n.r5p(chl_stroma),
        n.rubp(chl_stroma),
        n.ru5p(chl_stroma),
        n.atp(chl_stroma),
    ]

    model.add_derived(
        name=n.pi(chl_stroma),
        fn=_pi_cbb,
        args=args,
    )

    return model


def add_orthophosphate_moiety_cbb_pr(
    model: Model,
    *,
    chl_stroma: str = "",
    total: str | None = None,
) -> Model:
    total = static(model, n.total_orthophosphate(), 20.0) if total is None else total

    args = [
        total,
        n.pga(chl_stroma),
        n.bpga(chl_stroma),
        n.gap(chl_stroma),
        n.dhap(chl_stroma),
        n.fbp(chl_stroma),
        n.f6p(chl_stroma),
        n.g6p(chl_stroma),
        n.g1p(chl_stroma),
        n.sbp(chl_stroma),
        n.s7p(chl_stroma),
        n.e4p(chl_stroma),
        n.x5p(chl_stroma),
        n.r5p(chl_stroma),
        n.rubp(chl_stroma),
        n.ru5p(chl_stroma),
        n.atp(chl_stroma),
        n.pgo(chl_stroma),
    ]

    model.add_derived(
        name=n.pi(chl_stroma),
        fn=_pi_cbb,
        args=args,
    )

    return model


def _rate_fluorescence(
    Q: float,
    B0: float,
    B2: float,
    ps2cs: float,
    k2: float,
    kF: float,
    kH: float,
) -> float:
    return ps2cs * kF * B0 / (kF + k2 + kH * Q) + ps2cs * kF * B2 / (kF + kH * Q)


def add_readouts(
    model: Model,
    *,
    pq: bool = False,
    fd: bool = False,
    pc: bool = False,
    nadph: bool = False,
    atp: bool = False,
    fluorescence: bool = False,
) -> Model:
    if pq:
        model.add_readout(
            name="PQ_ox/tot",
            fn=div,
            args=[n.pq_red(), n.total_pq()],
        )
    if fd:
        model.add_readout(
            name="Fd_ox/tot",
            fn=div,
            args=[n.fd_red(), n.total_ferredoxin()],
        )
    if pc:
        model.add_readout(
            name="PC_ox/tot",
            fn=div,
            args=[n.pc_red(), n.total_pc()],
        )
    if nadph:
        model.add_readout(
            name="NADPH/tot",
            fn=div,
            args=[n.nadph(), n.total_nadp()],
        )
    if atp:
        model.add_readout(
            name="ATP/tot",
            fn=div,
            args=[n.atp(), n.total_adenosines()],
        )
    if fluorescence:
        model.add_readout(
            name=n.fluorescence(),
            fn=_rate_fluorescence,
            args=[
                n.quencher(),
                n.b0(),
                n.b2(),
                n.ps2cs(),
                "k2",
                "kF",
                "kH",
            ],
        )
    return model


def _quencher(
    Psbs: float,
    Vx: float,
    Psbsp: float,
    Zx: float,
    y0: float,
    y1: float,
    y2: float,
    y3: float,
    kZSat: float,
) -> float:
    """co-operative 4-state quenching mechanism
    gamma0: slow quenching of (Vx - protonation)
    gamma1: fast quenching (Vx + protonation)
    gamma2: fastest possible quenching (Zx + protonation)
    gamma3: slow quenching of Zx present (Zx - protonation)
    """
    ZAnt = Zx / (Zx + kZSat)
    return y0 * Vx * Psbs + y1 * Vx * Psbsp + y2 * ZAnt * Psbsp + y3 * ZAnt * Psbs


def add_quencher(model: Model) -> Model:
    model.add_parameter("gamma0", 0.1)
    model.add_parameter("gamma1", 0.25)
    model.add_parameter("gamma2", 0.6)
    model.add_parameter("gamma3", 0.15)
    model.add_parameter("kZSat", 0.12)
    model.add_derived(
        name=n.quencher(),
        fn=_quencher,
        args=[
            n.psbs_de(),
            n.vx(),
            n.psbs_pr(),
            n.zx(),
            "gamma0",
            "gamma1",
            "gamma2",
            "gamma3",
            "kZSat",
        ],
    )
    return model


def _ph_lumen(protons: float) -> float:
    return -math.log10(protons * 0.00025)


def _dg_ph(r: float, t: float) -> float:
    return math.log(10) * r * t


def add_ph_lumen(model: Model, *, chl_lumen: str) -> Model:
    model.add_derived("dG_pH", _dg_ph, args=["R", "T"])

    model.add_derived(
        name=n.ph(chl_lumen),
        fn=_ph_lumen,
        args=[
            n.h(chl_lumen),
        ],
    )
    return model
