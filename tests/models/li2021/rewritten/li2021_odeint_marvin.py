from collections.abc import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from tqdm import tqdm


def _light_per_L(par: float):
    return 0.84 * par / 0.7


def _driving_force_Cl(Cl_stroma: float, Cl_lumen: float, Dy: float):
    return 0.06 * np.log10(Cl_stroma / Cl_lumen) + Dy


def calc_PsbS_Protonation(pH_lumen: float, pKPsbS: float):
    return 1 / (10 ** (3 * (pH_lumen - pKPsbS)) + 1)


def calc_NPQ(Z, PsbS_H, max_NPQ):
    return 0.4 * max_NPQ * PsbS_H * Z + 0.5 * max_NPQ * PsbS_H + 0.1 * max_NPQ * Z


def calc_phi2(NPQ, QA):
    return 1 / (1 + (1 + NPQ) / (4.88 * QA))


def calc_h(pH):
    return 10 ** (-1 * pH)


def calc_pmf(Dy, pH_lumen, pH_stroma):
    return Dy + 0.06 * (pH_stroma - pH_lumen)


def calc_kCBB(PAR):
    return 60 * (PAR / (PAR + 250))


def _v_PSII_recombination(Dy, QAm, pH_lumen, k_recomb):  # correct
    delta_delta_g_recomb = Dy + 0.06 * (7.0 - pH_lumen)
    return k_recomb * QAm * 10 ** (delta_delta_g_recomb / 0.06)


def _v_PSII_charge_separations(antenna_size, light_per_L, Phi2):  # correct
    return antenna_size * light_per_L * Phi2


def _v_b6f(
    pH_lumen,
    PQH2,
    PQ,
    PC_ox,
    PC_red,
    pKreg,
    b6f_content,
    Em7_PC,
    Em7_PQH2,
    pmf,
    max_b6f,
):  # correct
    pHmod = 1 - (1 / (10 ** (pH_lumen - pKreg) + 1))
    b6f_deprot = pHmod * b6f_content

    Em_PC = Em7_PC
    Em_PQH2 = Em7_PQH2 - 0.06 * (pH_lumen - 7.0)

    Keq_b6f = 10 ** ((Em_PC - Em_PQH2 - pmf) / 0.06)
    k_b6f = b6f_deprot * max_b6f

    k_b6f_reverse = k_b6f / Keq_b6f
    f_PQH2 = PQH2 / (
        PQH2 + PQ
    )  # want to keep the rates in terms of fraction of PQHs, not total number
    f_PQ = 1 - f_PQH2
    return f_PQH2 * PC_ox * k_b6f - f_PQ * PC_red * k_b6f_reverse


def _v_NDH(Fd_red, PQ, Fd_ox, PQH2, pH_stroma, Em7_PQH2, Em_Fd, k_NDH, pmf):
    Em_PQH2 = Em7_PQH2 - 0.06 * (pH_stroma - 7.0)
    deltaEm = Em_PQH2 - Em_Fd
    Keq_NDH = 10 ** ((deltaEm - pmf * 2) / 0.06)
    k_NDH_reverse = k_NDH / Keq_NDH
    return k_NDH * Fd_red * PQ - k_NDH_reverse * Fd_ox * PQH2


def _v_PGR(Fd_red, PQ, PQH2, PGR_vmax):
    return PGR_vmax * (Fd_red**4 / (Fd_red**4 + 0.1**4)) * PQ / (PQ + PQH2)


def _v_PSI_charge_separation(Fd_ox, P700_red, PSI_antenna_size, light_per_L):
    return P700_red * light_per_L * PSI_antenna_size * Fd_ox


def _v_PQ_reduction_QA(QAm, PQ, kQA):
    return QAm * PQ * kQA


def _v_PQH2_oxidation_QA(PQH2, QA, kQA, Keq_QA_PQ):
    return PQH2 * QA * kQA / Keq_QA_PQ


def _v_PC_oxidation_P700(PC_red, P700_ox, k_PC_to_P700):
    return PC_red * k_PC_to_P700 * P700_ox


def _v_LEF(Fd_red, NADP_pool, k_Fd_to_NADP):
    return k_Fd_to_NADP * NADP_pool * Fd_red


def _v_Mehler(Fd_red, Fd_ox):
    return 4 * 0.000265 * Fd_red / (Fd_red + Fd_ox)


def _v_CBB_NADPH(NADPH_pool, NADP_pool, t, k_CBC):
    # print(f"{NADPH_pool=}, {NADP_pool=}, {t=}, {k_CBC=}")
    return (
        k_CBC
        * (1.0 - np.exp(-t / 600))
        * (np.log(NADPH_pool / NADP_pool) - np.log(1.25))
        / (np.log(3.5 / 1.25))
    )


def _v_KEA(QAm, pH_lumen, K_lumen, H_lumen, H_stroma, K_stroma, k_KEA):
    qL = 1 - QAm
    qL_act = qL**3 / (qL**3 + 0.15**3)
    pH_act = 1 / (10 ** (1 * (pH_lumen - 6.0)) + 1)
    f_KEA_act = qL_act * pH_act
    return k_KEA * (H_lumen * K_stroma - H_stroma * K_lumen) * f_KEA_act


def _v_K_channel(K_lumen, Dy, K_stroma, perm_K):
    K_deltaG = -0.06 * np.log10(K_stroma / K_lumen) + Dy
    return perm_K * K_deltaG * (K_lumen + K_stroma) / 2


def _v_VCCN1(Cl_lumen, Cl_stroma, driving_force_Cl, k_VCCN1):
    relative_Cl_flux = (
        332 * (driving_force_Cl**3)
        + 30.8 * (driving_force_Cl**2)
        + 3.6 * driving_force_Cl
    )
    return k_VCCN1 * relative_Cl_flux * (Cl_stroma + Cl_lumen) / 2


def _v_CLCE(Cl_lumen, Cl_stroma, H_lumen, H_stroma, driving_force_Cl, pmf, k_CLCE):
    return (
        k_CLCE
        * (driving_force_Cl * 2 + pmf)
        * (Cl_stroma + Cl_lumen)
        * (H_lumen + H_stroma)
        / 4
    )


def _v_leak(H_lumen, pmf, k_leak):
    return pmf * k_leak * H_lumen


def _v_pmf_protons_activity(t, pmf, n, ATP_synthase_max_turnover, light_per_L):
    x = t / 165
    actvt = 0.2 + 0.8 * (x**4 / (x**4 + 1))
    v_proton_active = 1 - (
        1 / (10 ** ((pmf - 0.132) * 1.5 / 0.06) + 1)
    )  # reduced ATP synthase
    v_proton_inert = 1 - (
        1 / (10 ** ((pmf - 0.204) * 1.5 / 0.06) + 1)
    )  # oxidized ATP synthase

    v_active = actvt * v_proton_active * n * ATP_synthase_max_turnover
    v_inert = (1 - actvt) * v_proton_inert * n * ATP_synthase_max_turnover

    v_proton_ATP = v_active + v_inert

    if light_per_L > 0:
        return v_proton_ATP
    else:
        return 0


def _v_ZE(Z, kZE):
    return Z * kZE


def _v_VDE(V, pH_lumen, VDE_Hill, pKvde, VDE_max_turnover_number):
    pHmod = 1 / (10 ** (VDE_Hill * (pH_lumen - pKvde)) + 1)
    return V * VDE_max_turnover_number * pHmod


def neg_2_div(x: float, y: float):
    return -2 * x / y


def neg_point_one_val(x: float):
    return -0.1 * x


def neg_point_two_val(x: float):
    return -0.2 * x


def neg_thrice(x: float):
    return x * -3


def moeity(total, x):
    return total - x


def get_params() -> dict[str, float]:
    return {
        "PAR": 50,
        "k_recomb": 0.33,
        "triplet_yield": 0.45,
        "triplet_to_singletO2_yield": 1,
        "PSII_antenna_size": 0.5,
        "b6f_content": 0.433,
        "pKreg": 6.2,
        "Em7_PC": 0.37,
        "Em7_PQH2": 0.11,
        "max_b6f": 300,
        "pKPsbS": 6.2,
        "max_NPQ": 3,
        "pH_stroma": 7.8,
        "Em_Fd": -0.42,
        "k_NDH": 1000,
        "PGR_vmax": 0,
        "PSI_antenna_size": 0.5,
        "kQA": 1000,
        "Keq_QA_PQ": 200,
        "k_PC_to_P700": 5000,
        "k_Fd_to_NADP": 1000,
        "k_CBC": 60,
        "K_stroma": 0.1,
        "k_KEA": 2500000,
        "perm_K": 150,
        "lumen_protons_per_turnover": 0.000587,
        "k_VCCN1": 12,
        "k_CLCE": 800000,
        "n": 14 / 3,
        "ATP_synthase_max_turnover": 200,
        "buffering_capacity": 0.014,
        "Volts_per_charge": 0.047,
        "kZE": 0.004,
        "VDE_Hill": 4,
        "pKvde": 5.65,
        "VDE_max_turnover_number": 0.08,
        "k_leak": 3e7,
        # Moeity
        "QA_total": 1,
        "PQ_total": 7,
        "P700_total": 0.667,
        "PC_total": 2,
        "Fd_total": 1,
        "NA_total": 5,
        "X_total": 1,
    }


def get_derived_params(p: dict[str, float]) -> dict[str, float]:
    return {
        "light_per_L": _light_per_L(p["PAR"]),
        "k_CBC": calc_kCBB(500), # CHANGED
    }


def _get_derived(
    t: float,
    y: Iterable[float],
    p: dict[str, float],
) -> Iterable[float]:
    (
        QAm,
        PQH2,
        pH_lumen,
        Dy,
        K_lumen,
        PC_ox,
        P700_ox,
        Zx,
        singletO2,
        Fd_red,
        NADPH,
        Cl_lumen,
        Cl_stroma,
    ) = y

    PsbS_H = calc_PsbS_Protonation(pH_lumen, p["pKPsbS"])
    NPQ = calc_NPQ(Zx, PsbS_H, p["max_NPQ"])
    QA = moeity(p["QA_total"], QAm)
    return (
        PsbS_H,
        NPQ,
        QA,
        calc_phi2(NPQ, QA),
        moeity(p["PQ_total"], PQH2),
        moeity(p["P700_total"], P700_ox),
        moeity(p["PC_total"], PC_ox),
        moeity(p["Fd_total"], Fd_red),
        moeity(p["NA_total"], NADPH),
        calc_h(pH_lumen),
        calc_h(p["pH_stroma"]),
        moeity(p["X_total"], Zx),
        calc_pmf(Dy, pH_lumen, p["pH_stroma"]),
        _driving_force_Cl(Cl_stroma, Cl_lumen, Dy),
    )


def get_derived(t: float, y: Iterable[float], p: dict[str, float]) -> dict[str, float]:
    return dict(
        zip(
            [
                "PsbS_H",
                "NPQ",
                "QA",
                "Phi2",
                "PQ",
                "P700_red",
                "PC_red",
                "Fd_ox",
                "NADP",
                "H_lumen",
                "H_stroma",
                "V",
                "pmf",
                "driving_force_Cl",
            ],
            _get_derived(t, y, p),
            strict=True,
        )
    )


def _get_rates(t, y, d, p, dp) -> Iterable[float]:
    (
        QAm,
        PQH2,
        pH_lumen,
        Dy,
        K_lumen,
        PC_ox,
        P700_ox,
        Zx,
        singletO2,
        Fd_red,
        NADPH,
        Cl_lumen,
        Cl_stroma,
    ) = y
    (
        PsbS_H,
        NPQ,
        QA,
        Phi2,
        PQ,
        P700_red,
        PC_red,
        Fd_ox,
        NADP,
        H_lumen,
        H_stroma,
        V,
        pmf,
        driving_force_Cl,
    ) = d

    v_PSII_recombination = _v_PSII_recombination(Dy, QAm, pH_lumen, p["k_recomb"])
    v_PSII_charge_separations = _v_PSII_charge_separations(
        p["PSII_antenna_size"], dp["light_per_L"], Phi2
    )
    v_PQ_reduction_QA = _v_PQ_reduction_QA(QAm, PQ, p["kQA"])  # Replaced loose quation
    v_PQH2_oxidation_QA = _v_PQH2_oxidation_QA(
        PQH2, QA, p["kQA"], p["Keq_QA_PQ"]
    )  # Replaced loose quation
    v_b6f = _v_b6f(
        pH_lumen,
        PQH2,
        PQ,
        PC_ox,
        PC_red,
        p["pKreg"],
        p["b6f_content"],
        p["Em7_PC"],
        p["Em7_PQH2"],
        pmf,
        p["max_b6f"],
    )
    v_NDH = _v_NDH(
        Fd_red,
        PQ,
        Fd_ox,
        PQH2,
        p["pH_stroma"],
        p["Em7_PQH2"],
        p["Em_Fd"],
        p["k_NDH"],
        pmf,
    )
    v_PGR = _v_PGR(Fd_red, PQ, PQH2, p["PGR_vmax"])
    v_PSI_charge_separation = _v_PSI_charge_separation(
        Fd_ox, P700_red, p["PSI_antenna_size"], dp["light_per_L"]
    )
    v_PC_oxidation_P700 = _v_PC_oxidation_P700(
        PC_red, P700_ox, p["k_PC_to_P700"]
    )  # Replaced loose quation
    v_LEF = _v_LEF(Fd_red, NADP, p["k_Fd_to_NADP"])  # Replaced dLEF
    v_Mehler = _v_Mehler(Fd_red, Fd_ox)
    v_CBB_NADPH = _v_CBB_NADPH(NADPH, NADP, t, dp["k_CBC"])
    v_KEA = _v_KEA(QAm, pH_lumen, K_lumen, H_lumen, H_stroma, p["K_stroma"], p["k_KEA"])
    v_K_channel = _v_K_channel(K_lumen, Dy, p["K_stroma"], p["perm_K"])
    v_VCCN1 = _v_VCCN1(Cl_lumen, Cl_stroma, driving_force_Cl, p["k_VCCN1"])
    v_CLCE = _v_CLCE(
        Cl_lumen, Cl_stroma, H_lumen, H_stroma, driving_force_Cl, pmf, p["k_CLCE"]
    )
    v_leak = _v_leak(H_lumen, pmf, p["k_leak"])
    v_pmf_protons_activity = _v_pmf_protons_activity(
        t, pmf, p["n"], p["ATP_synthase_max_turnover"], dp["light_per_L"]
    )
    v_ZE = _v_ZE(Zx, p["kZE"])
    v_VDE = _v_VDE(V, pH_lumen, p["VDE_Hill"], p["pKvde"], p["VDE_max_turnover_number"])

    return (
        v_PSII_recombination,
        v_PSII_charge_separations,
        v_PQ_reduction_QA,
        v_PQH2_oxidation_QA,
        v_b6f,
        v_NDH,
        v_PGR,
        v_PSI_charge_separation,
        v_PC_oxidation_P700,
        v_LEF,
        v_Mehler,
        v_CBB_NADPH,
        v_KEA,
        v_K_channel,
        v_VCCN1,
        v_CLCE,
        v_leak,
        v_pmf_protons_activity,
        v_ZE,
        v_VDE,
    )


def get_rates(t: float, y: Iterable[float], p: dict[str, float]) -> dict[str, float]:
    return dict(
        zip(
            [
                "v_PSII_recombination",
                "v_PSII_charge_separations",
                "v_PQ_reduction_QA",
                "v_PQH2_oxidation_QA",
                "v_b6f",
                "v_NDH",
                "v_PGR",
                "v_PSI_charge_separation",
                "v_PC_oxidation_P700",
                "v_LEF",
                "v_Mehler",
                "v_CBB_NADPH",
                "v_KEA",
                "v_K_channel",
                "v_VCCN1",
                "v_CLCE",
                "v_leak",
                "v_pmf_protons_activity",
                "v_ZE",
                "v_VDE",
            ],
            _get_rates(t, y, _get_derived(t, y, p), p, dp=get_derived_params(p)),
            strict=True,
        )
    )


def model_old(t, y, p):
    d = _get_derived(t, y, p)
    (
        v_PSII_recombination,
        v_PSII_charge_separations,
        v_PQ_reduction_QA,
        v_PQH2_oxidation_QA,
        v_b6f,
        v_NDH,
        v_PGR,
        v_PSI_charge_separation,
        v_PC_oxidation_P700,
        v_LEF,
        v_Mehler,
        v_CBB_NADPH,
        v_KEA,
        v_K_channel,
        v_VCCN1,
        v_CLCE,
        v_leak,
        v_pmf_protons_activity,
        v_ZE,
        v_VDE,
    ) = _get_rates(t, y, d, p, dp=get_derived_params(p))

    # ODEs
    dsingletO2 = (
        p["triplet_yield"] * p["triplet_to_singletO2_yield"]
    ) * v_PSII_recombination
    dQAm = (
        v_PSII_charge_separations
        + v_PQH2_oxidation_QA
        - v_PQ_reduction_QA
        - v_PSII_recombination
    )
    dPQH2 = (
        0.5 * v_PQ_reduction_QA
        + 0.5 * v_NDH
        + 0.5 * v_PGR
        - 0.5 * v_b6f
        - 0.5 * v_PQH2_oxidation_QA
    )
    dP700_ox = v_PSI_charge_separation - v_PC_oxidation_P700
    dPC_ox = v_PC_oxidation_P700 - v_b6f
    dFd_red = v_PSI_charge_separation - v_LEF - v_NDH - v_PGR - v_Mehler
    dNADPH = 0.5 * v_LEF - v_CBB_NADPH
    dK_lumen = (v_KEA - v_K_channel) * p["lumen_protons_per_turnover"]
    dCl_lumen = (v_VCCN1 + 2 * v_CLCE) * p["lumen_protons_per_turnover"]
    dCl_stroma = -0.1 * (v_VCCN1 + 2 * v_CLCE) * p["lumen_protons_per_turnover"]

    l_per_b = p["lumen_protons_per_turnover"] / p["buffering_capacity"]
    dpH_lumen = (
        -l_per_b * v_PSII_charge_separations
        + l_per_b * v_PSII_recombination
        - 2 * l_per_b * v_b6f
        - 2 * l_per_b * v_NDH
        + l_per_b * v_pmf_protons_activity
        + l_per_b * v_leak
        + l_per_b * v_KEA
        + l_per_b * v_CLCE
    )
    dDy = (
        v_PSII_charge_separations
        - v_PSII_recombination
        + v_PSI_charge_separation
        + v_b6f
        + 2 * v_NDH
        - v_K_channel
        - v_pmf_protons_activity
        - v_leak
        - v_VCCN1
        - 3 * v_CLCE
    ) * p["Volts_per_charge"]
    dZx = v_VDE - v_ZE

    return (
        dQAm,
        dPQH2,
        dpH_lumen,
        dDy,
        dK_lumen,
        dPC_ox,
        dP700_ox,
        dZx,
        dsingletO2,
        dFd_red,
        dNADPH,
        dCl_lumen,
        dCl_stroma,
    )


def add_readouts(out: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    out["delta_pH"] = params["pH_stroma"] - out.loc[:, "pH_lumen"]
    out["delta_pH_V"] = out.loc[:, "delta_pH"] * 0.06
    return out


def simulate(
    model: Callable,
    y0: Iterable[float],
    t: Iterable[float],
    variables: list[str],
    params: dict[str, float],
) -> pd.DataFrame:
    concs = pd.DataFrame(
        dict(
            zip(
                variables,
                odeint(
                    model,
                    y0=y0,
                    t=t,
                    args=(params,),
                    tfirst=True,
                    rtol=1e-6,
                    atol=1e-6,
                ).T,
                strict=True,
            ),
        ),
        index=t,
    )

    derived = pd.DataFrame(
        {t: get_derived(t, y.to_numpy(), params) for t, y in concs.iterrows()}
    ).T

    fluxes = pd.DataFrame(
        {t: get_rates(t, y.to_numpy(), params) for (t, y) in concs.iterrows()}
    ).T

    return add_readouts(pd.concat((concs, derived, fluxes), axis=1), params)


def lightsim_20min_5min(par: float, params: dict):
    labels_y0: list[tuple[str, float]] = [
        ("QAm", 0),
        ("PQH2", 0),
        ("pH_lumen", 7.8),
        ("Dy", 0),
        ("K_lumen", 0.1),
        ("PC_ox", 0),
        ("P700_ox", 0),
        ("Zx", 0),
        ("singletO2", 0),
        ("Fd_red", 0),
        ("NADPH", 1.5),
        ("Cl_lumen", 0.04),
        ("Cl_stroma", 0.04),
    ]
    variables: list[str] = [i[0] for i in labels_y0]
    y0: list[float] = [i[1] for i in labels_y0]

    out: list[pd.DataFrame] = []
    # First simulation
    params["PAR"] = par
    params.update(get_derived_params(params))
    out.append(
        simulate(
            model_old,
            y0,
            t=np.linspace(0, 1200, 1000),
            variables=variables,
            params=params,
        )
    )

    # Second simulation
    params["PAR"] = 0
    params.update(get_derived_params(params))
    out.append(
        simulate(
            model_old,
            out[-1].iloc[-1].loc[variables],
            t=np.linspace(1200, 1200 + 5 * 60, 1000),
            variables=variables,
            params=params,
        )
    )

    out_df = pd.concat(out, axis=0)
    out_df["time_sec"] = out_df.index
    out_df["time_min"] = out_df.index / 60
    out_df["time_h"] = out_df.index / 60 / 60

    return out_df


def change_params(base_params, change_str):
    if "clce2" in change_str:
        base_params["k_CLCE"] = 0
    if "vccn1" in change_str:
        base_params["k_VCCN1"] = 0
    if "kea3" in change_str:
        base_params["k_KEA"] = 0

    return base_params


def create_fig3(params: dict[str, float]):
    res_new = {}
    for g_type in (
        pbar := tqdm(
            [
                "WT",
                "clce2",
                "vccn1",
                "kea3",
                "clce2vccn1",
                "clce2kea3",
                "vccn1kea3",
                "vccn1clce2kea3",
            ]
        )
    ):
        pbar.set_description(f"Simulating '{g_type}'")
        if g_type != "WT":
            new_pars = change_params(params.copy(), g_type)
        else:
            new_pars = params.copy()
        res_new[g_type] = {
            "100": lightsim_20min_5min(100, new_pars),
            "500": lightsim_20min_5min(500, new_pars),
        }

    fig3, axs = plt.subplot_mosaic(
        [
            ["A"] * 7 + ["B"] * 7 + ["C"] * 7 + ["D"] * 7 + ["E"] * 7,
            ["F"] * 5
            + ["G"] * 5
            + ["H"] * 5
            + ["I"] * 5
            + ["J"] * 5
            + ["K"] * 5
            + ["L"] * 5,
        ],
        figsize=(25, 5),
    )

    for res, color in zip(
        [res_new["WT"]["100"], res_new["WT"]["500"]], ["black", "red"]
    ):
        axs["A"].plot(res["time_min"], 1 - res["QAm"], color=color)
        axs["B"].plot(res["time_min"], res["Phi2"], color=color)
        axs["C"].plot(res["time_min"], res["NPQ"], color=color)

    for ax in [axs["A"], axs["B"], axs["C"]]:
        ax.set_xlim(0, 25)
        ax.set_xlabel("Time (min)")
        ax.spines[["right", "top"]].set_visible(False)

    axs["A"].set_ylim(0, 1.2)
    axs["A"].set_yticks(np.arange(0, 1.4, 0.2))
    axs["A"].set_ylabel("qL")
    axs["A"].set_title("qL")

    axs["B"].set_ylim(0, 1.0)
    axs["B"].set_ylabel(r"$\mathrm{\phi}$II")
    axs["B"].set_title(r"$\mathrm{\phi}$II")

    axs["C"].set_ylim(0, 2.5)
    axs["C"].set_yticks(np.arange(0, 3, 1))
    axs["C"].set_ylabel("NPQ")
    axs["C"].set_title("NPQ")

    for ax_let, key in zip(
        list("FGHIJKL"),
        [
            "clce2",
            "vccn1",
            "kea3",
            "clce2vccn1",
            "clce2kea3",
            "vccn1kea3",
            "vccn1clce2kea3",
        ],
    ):
        ax = axs[ax_let]
        for par, color in zip(["100", "500"], ["black", "red"]):
            ax.plot(
                res_new[key][par]["time_min"],
                res_new[key][par]["NPQ"] - res_new["WT"][par]["NPQ"],
                color=color,
            )

        ax.set_ylim(-0.7, 0.7)
        ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
        ax.set_ylabel(r"$\Delta$ NPQ")
        ax.set_xlim(0, 25)
        ax.set_xticks([0, 5, 10, 15, 20, 25])

        title = ""
        if "clce2" in key:
            title += "c"
        if "vccn1" in key:
            title += "v"
        if "kea3" in key:
            title += "k"
        ax.set_title(title)

    x_coord = 0
    width = 0.125
    xcoords = []
    for key in [
        "WT",
        "clce2",
        "vccn1",
        "kea3",
        "clce2vccn1",
        "clce2kea3",
        "vccn1kea3",
        "vccn1clce2kea3",
    ]:
        axs["D"].scatter(
            x_coord - width,
            max(res_new[key]["100"]["delta_pH_V"])
            - max(res_new["WT"]["100"]["delta_pH_V"]),
            color="black",
        )
        axs["D"].scatter(
            x_coord + width,
            max(res_new[key]["500"]["delta_pH_V"])
            - max(res_new["WT"]["500"]["delta_pH_V"]),
            color="red",
        )

        max_Z = max(res_new[key]["500"]["Zx"])
        max_V = max(res_new[key]["500"]["Vx"])
        axs["E"].scatter(x_coord, max_Z / (max_Z + max_V), color="black")
        xcoords += [x_coord]
        x_coord += 2

    labels = []
    for i in [
        "WT",
        "clce2",
        "vccn1",
        "kea3",
        "clce2vccn1",
        "clce2kea3",
        "vccn1kea3",
        "vccn1clce2kea3",
    ]:
        label = ""
        if "clce2" in i:
            label += "c"
        if "vccn1" in i:
            label += "v"
        if "kea3" in i:
            label += "k"
        if label == "":
            label = i
        labels.append(label)

    for ax in [axs["D"], axs["E"]]:
        ax.set_xticks(xcoords, labels)
        ax.set_xlim(xcoords[0] - 0.5, xcoords[-1] + 0.5)
        ax.spines[["top"]].set_visible(False)

    axs["D"].plot([-10, x_coord + 10], [0, 0], zorder=-1, color="black", lw=0.5)
    axs["D"].set_ylim(-0.006, 0.006)
    axs["D"].set_yticks([-0.004, 0, 0.004])

    axs["E"].set_ylim(0, 0.7)

    plt.subplots_adjust(hspace=0.5, wspace=2000)

    return fig3, axs
