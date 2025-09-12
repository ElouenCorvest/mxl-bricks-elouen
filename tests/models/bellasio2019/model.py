from mxlpy import Model, Variable, Parameter, Derived
import mxlpy.units as units
import mxlbricks.names as n
from mxlbricks.derived import add_adenosin_moiety, add_nadp_moiety
from mxlbricks.fns import div, neg_one_div, two_div, one_div, neg_half_div, value, half_div
import numpy as np
from sympy.physics.units import bar

def _pi_bellasio2019(total, pga, dhap, ru5p, rubp, atp):
    return total - pga - dhap - ru5p - 2 * rubp - atp

def _Et(vmax_rub, kcat_rub, V_m):
    return (vmax_rub / kcat_rub) / V_m

def ract_gs_time_dependance(x, x_steady, inc, dec):
    if x < x_steady:
        return (x_steady - x) / inc
    else:
        return (x_steady - x) / dec

def _i20(pfd, s):
    return pfd * s

def _i10(i_20, y_ii_ll, y_i_ll):
    return i_20 * y_ii_ll / y_i_ll

def _chi(f_cyc, y_ii_ll):
    return f_cyc / (1 + f_cyc + y_ii_ll)

def _i1(chi, i10):
    return (1 + chi) * i10

def _f_cyc(j_atp, j_nadph, v_atp, v_nadph):
    return max(0, -1 + 15**(v_atp / j_atp - v_nadph / j_nadph))

def _i2(y_ii_ll, chi, i20):
    return (1 / y_ii_ll - chi) * i20 * y_ii_ll

def _y_ii(y_ii_ll, v_atp, j_atp, v_nadph, j_nadph, pfd, alpha, V0, theta):
    f_ppfd = non_rect_hyperbole(pfd, alpha, V0, theta)
    return y_ii_ll * (v_atp / j_atp) * (v_nadph / j_nadph) * (1 - max(0, f_ppfd))

def _j2(i2, y_ii):
    return i2 * y_ii

def _j1(j2, f_cyc):
    return j2 / 1 - f_cyc

def _f_pseudocyc(j_nadph, o2, v_nadph, f_pseudocycNR):
    return f_pseudocycNR + 4 * o2 * (1 - v_nadph / j_nadph)

def _j_nadph_steady(j1, f_cyc, f_pseudocyc):
    top = 1 - f_cyc - f_pseudocyc
    bottom = 2
    return j1 * top / bottom

def _j_atp_steady(j2, j1, f_cyc, fq, f_ndh, h):
    jcyt = (1 - fq) * j1
    jq = fq * j1
    jndh = f_cyc * f_ndh * j1
    
    return (j2 + jcyt + 2 * jq + 2 * jndh) / h

def _gs_steady(tau0, f_rubp, chi_beta, phi, pi_e, Kh, Ds, gs0):

    tau = tau0 + f_rubp
    top = chi_beta * tau * (phi + pi_e)
    bottom = 1 + chi_beta * tau * (1 / Kh) * Ds
    
    return max(gs0, top / bottom)

def atp_nadph_time_dependance(j_x, j_x_steady, kj_x):
    if j_x < j_x_steady:
        return (j_x_steady - j_x) / kj_x
    else:
        return j_x_steady
        
def non_rect_hyperbole(x, alpha, V0, theta):
    # print(np.sqrt((alpha * x + 1 - V0)**2 - 4 * alpha * x * theta))
    # top = alpha * x + 1 - V0 - np.sqrt((alpha * x + 1 - V0)**2 - 4 * alpha * x * theta)
    # bottom = 2 * theta
    return (alpha * x + 1 - V0) / (2*theta) - np.sqrt((alpha * x + 1 - V0)**2 - 4 * alpha * x * theta * (1-V0)) / (2* theta) + V0

def _Ract_eq(co2, ppfd, alpha_ppfd, V0_ppfd, theta_ppfd, alpha_co2, V0_co2, theta_co2):
    f_ppfd = non_rect_hyperbole(ppfd, alpha_ppfd, V0_ppfd, theta_ppfd)
    f_co2 = non_rect_hyperbole(co2, alpha_co2, V0_co2, theta_co2)
    return f_ppfd * f_co2

def _f_rubp(rubp, Et, k_extra_rubp):
    top = Et + k_extra_rubp + rubp - np.sqrt((Et + k_extra_rubp + rubp)**2 - 4 * rubp * Et)
    bottom = 2 * Et
    return top / bottom

def _co2_diss(ci, co2, gm, Kh_co2):
    return (gm * (ci - co2 * Kh_co2)) / 1000

def _stom_diff(ci, gs, ca):
    return (gs * (ca - ci)) / 1000

def _km_rubp_extra(pga, nadp, adp, pi, km_rubp, ki_pga, ki_nadp, ki_adp, ki_pi):
    return km_rubp * (1 + pga / ki_pga + nadp / ki_nadp + adp / ki_adp + pi / ki_pi)

def _rubisco_carboxylation_bellasio(rubp, co2, Ract, km_co2, o2, km_o2, vmax_rc, f_rubp, k_extra_rubp):
    k_extra_co2 = km_co2 * (1 + o2 / km_o2)
    
    top = vmax_rc * Ract * f_rubp * rubp * co2
    bottom = (k_extra_co2 + co2) * (k_extra_rubp + rubp)
    
    return top / bottom

def _rubisco_oxygenase_bellasio(co2, o2, S_co, v_c):
    gamma_star = 1 / (2* S_co)
    return v_c * 2 * gamma_star * o2 * co2

def _prkase(atp, rubp, ru5p, pga, adp, pi, vmax, k_eq, km_atp, ki_adp, km_ru5p, ki_pga, ki_rubp, ki_pi):
    top = vmax * atp * ru5p - (atp * rubp) / k_eq
    bottom = (atp + km_atp * (1 + adp / ki_adp)) * (ru5p + km_ru5p * (1 + pga / ki_pga + rubp / ki_rubp + pi / ki_pi))
    return top / bottom

def _v_pgareduction(atp, pga, nadph, adp, vmax, km_atp, km_pga, km_nadph, ki_adp):
    top = vmax * atp * pga * nadph
    bottom = (pga + km_pga * (1 + adp / ki_adp)) * (atp + km_atp * (1+ adp / ki_adp)) * (nadph + km_nadph * (1 + adp / ki_adp))
    return top / bottom

def _v_carbohydrate_synthesis(dhap, pi, adp, vmax, v_pgareduction, keq, km_dhap, ki_adp):
    top = vmax * (dhap - 0.4) * (1 - np.abs(v_pgareduction) * pi / keq)
    bottom = dhap + km_dhap * (1 + adp / ki_adp)
    return top / bottom

def _v_rpp(dhap, ru5p, vmax, k_eq, km_dhap):
    top = vmax * dhap * (1 - ru5p / k_eq)
    bottom = dhap + km_dhap
    return top / bottom

def _v_co2_hydration(co2, hco3, proton, vmax, k_eq, km_co2, km_hco3):
    top = vmax * (co2 - hco3 * proton / k_eq)
    bottom = km_co2 * (1 + co2 / km_co2 + hco3 / km_hco3)
    return top / bottom

def _v_NADPH(nadph, nadp, j_nadph, k_eq, km_nadp, km_nadph):
    top = j_nadph * (nadp - nadph/k_eq)
    bottom = km_nadp * (1 + nadp / km_nadp + nadph/km_nadph)
    return top / bottom

def _v_atp(atp, adp, pi, j_atp, k_eq, km_adp, km_pi, km_atp):
    top = j_atp * (atp * pi - adp / k_eq)
    bottom = km_adp * km_pi * (1 + adp / km_adp + atp / km_atp + pi / km_pi + atp * pi / (km_adp * km_pi))
    return top / bottom

def neg_fivethirds_div(x):
    return -(5/3) * (1 / x)

def neg_onethirds_div(x):
    return -(1/3) * (1 / x)

def ci_initial(ca):
    return 0.65 * ca

def get_bellasio2019(Ca=400, Kh_co2=30303) -> Model:
    """
    'A generalised dynamic model of leaf-level C3 photosynthesis combining light and dark reactions with stomatal behaviour'
    
    Chandra Bellasio
    
    Photosynthesis research, 2019 (https://doi.org/10.1007/s11120-018-0601-1)
    """
    
    model = Model()
    
    unit_mM = units.mmol / units.liter
    
    model.add_variables({
        n.co2(): Variable(0.3 * Ca / Kh_co2, unit_mM),
        "HCO3": Variable(0.1327, unit_mM),
        n.rubp(): Variable(2, unit_mM),
        n.pga(): Variable(4, unit_mM),
        n.dhap(): Variable(4, unit_mM),
        n.atp(): Variable(0.68, unit_mM),
        n.nadph(): Variable(0.21, unit_mM),
        n.ru5p(): Variable(0.34, unit_mM),
        "Ract": Variable(1),
        "J_NADPH": Variable(0.1, unit_mM), # Check units
        "J_ATP": Variable(0.16, unit_mM), # Check units
        "Ci": Variable(0.65 * Ca),
        "gs": Variable(0.25)
    })
    
    model.add_parameters({
        # Moeities
        n.total_adenosines(): Parameter(1.5, unit_mM),
        n.total_orthophosphate(): Parameter(15, unit_mM),
        # Overall Params
        "p_o2": Parameter(210000, units.micro * bar),
        "Kh_o2": Parameter(833300, units.micro * bar / unit_mM),
        "V_m": Parameter(0.03, units.liter / units.sqm),
        n.pfd(): Parameter(1500, units.ppfd),
        "RLight": Parameter(0.001, units.mmol / (units.sqm * units.second)),
        "s": Parameter(0.43),
        "Y(II)_LL": Parameter(0.72),
        "Y(I)_LL": Parameter(1),
        "alpha_ppfd_Y(II)": Parameter(0.00125),
        "V0_ppfd_Y(II)": Parameter(-0.8),
        "theta_ppfd_Y(II)": Parameter(0.7),
        "f_pseudocycNR": Parameter(0.01),
        "fq": Parameter(1),
        "f_ndh": Parameter(0),
        "h": Parameter(4),
        "Ca": Parameter(Ca),
        # Rubisco activation
        "alpha_ppfd_rub": 0.0018,
        "V0_ppfd_rub": 0.16,
        "theta_ppfd_rub": 0.95,
        "alpha_co2": 400,
        "V0_co2": -0.2,
        "theta_co2": 0.98,
        "tau_i": Parameter(360, units.second),
        "tau_d": Parameter(1200, units.second),
        # Rubisco Carboxylation
        n.km(n.rubisco_carboxylase(), n.co2()): Parameter(0.014, unit_mM),
        n.km(n.rubisco_carboxylase(), n.rubp()): Parameter(0.02, unit_mM),
        n.km(n.rubisco_carboxylase(), n.o2()): Parameter(0.222, unit_mM),
        n.ki(n.rubisco_carboxylase(), n.pga()): Parameter(2.52, unit_mM),
        n.ki(n.rubisco_carboxylase(), n.nadp()): Parameter(0.21, unit_mM), # Wrong name in paper
        n.ki(n.rubisco_carboxylase(), n.adp()): Parameter(0.2, unit_mM),
        n.ki(n.rubisco_carboxylase(), n.pi()): Parameter(3.6, unit_mM),
        n.vmax(n.rubisco_carboxylase()): Parameter(0.2, units.mmol / (units.sqm * units.second)),
        n.kcat(n.rubisco_carboxylase()): Parameter(4.7, units.per_second),
        # Rubisco Oxygenation
        "S_co": Parameter(2200),
        # RuP_phosp
        n.vmax(n.r1p_kinase()): Parameter(1.17, units.mmol / (units.sqm * units.second)), 
        n.keq(n.r1p_kinase()): Parameter(6846, unit_mM), 
        n.km(n.r1p_kinase(), n.atp()): Parameter(0.625, unit_mM), 
        n.ki(n.r1p_kinase(), n.adp()): Parameter(0.1, unit_mM), 
        n.km(n.r1p_kinase(), n.ru5p()): Parameter(0.034, unit_mM), 
        n.ki(n.r1p_kinase(), n.pga()): Parameter(2, unit_mM), 
        n.ki(n.r1p_kinase(), n.rubp()): Parameter(0.7, unit_mM), 
        n.ki(n.r1p_kinase(), n.pi()): Parameter(4, unit_mM),
        # PR
        n.vmax("v_pgareduction"): Parameter(0.4, units.mmol / (units.sqm * units.second)),
        n.km("v_pgareduction", n.atp()): Parameter(0.3, unit_mM),
        n.km("v_pgareduction", n.pga()): Parameter(10, unit_mM),
        n.km("v_pgareduction", n.nadph()): Parameter(0.05, unit_mM),
        n.ki("v_pgareduction", n.adp()): Parameter(0.89, unit_mM),
        # CS
        n.vmax("v_carbohydrate_synthesis"): Parameter(1, units.mmol / (units.sqm * units.second)),
        n.keq("v_carbohydrate_synthesis"): Parameter(0.8),
        n.km("v_carbohydrate_synthesis", n.dhap()): Parameter(22, unit_mM),
        n.ki("v_carbohydrate_synthesis", n.adp()): Parameter(0.3, unit_mM),
        # RPP
        n.vmax("v_rpp"): Parameter(0.0585, units.mmol / (units.sqm * units.second)),
        n.keq("v_rpp"): Parameter(0.06),
        n.km("v_rpp", n.dhap()): Parameter(0.5, unit_mM),
        # CA
        n.h(): Parameter(5.012e-5, unit_mM),
        n.vmax("v_co2_hydration"): Parameter(200, units.mmol / (units.sqm * units.second)),
        n.keq("v_co2_hydration"): Parameter(0.00056, unit_mM),
        n.km("v_co2_hydration", n.co2()): Parameter(2.8, unit_mM),
        n.km("v_co2_hydration", n.hco3()): Parameter(34, unit_mM),
        # v_NADPH
        n.keq("v_NADPH"): Parameter(502, unit_mM),
        n.km("v_NADPH", n.nadp()): Parameter(0.0072, unit_mM),
        n.km("v_NADPH", n.nadph()): Parameter(0.036, unit_mM),
        "Kj_NADPH": Parameter(200, units.second),
        # v_ATP
        n.keq("v_ATP"): Parameter(5734),
        n.km("v_ATP", n.adp()): Parameter(0.014, unit_mM),
        n.km("v_ATP", n.pi()): Parameter(0.3, unit_mM),
        n.km("v_ATP", n.atp()): Parameter(0.3, unit_mM),
        "Kj_ATP": Parameter(200, units.second),
        # CO2 Dissolution
        "gm": Parameter(0.5, units.mol / (units.sqm * units.second)),
        "Kh_co2": Parameter(Kh_co2, units.micro * bar / unit_mM),
        # Stomata
        "Kd": Parameter(150, units.second),
        "Ki": Parameter(900, units.second),
        "tau0": Parameter(-0.1, unit_mM),
        "chi_beta": Parameter(0,5),
        "phi": Parameter(0),
        "pi_e": Parameter(1.2),
        "Kh": Parameter(12),
        "Ds": Parameter(10),
        "gs0": Parameter(0.01),
    })
    
    # Derived
    add_adenosin_moiety(model, total=n.total_adenosines())
    add_nadp_moiety(model)
    model.add_derived(
        n.pi(),
        fn=_pi_bellasio2019,
        args=[
            n.total_orthophosphate(),
            n.pga(),
            n.dhap(),
            n.ru5p(),
            n.rubp(),
            n.atp()
        ]
    )
    model.add_derived(
        "Et",
        fn=_Et,
        args=[n.vmax(n.rubisco_carboxylase()), n.kcat(n.rubisco_carboxylase()), "V_m"]
    )
    model.add_derived(
        n.km(n.rubisco_carboxylase(), f"{n.rubp()}_extra"),
        fn=_km_rubp_extra,
        args=[n.pga(), n.nadp(), n.adp(), n.pi(), n.km(n.rubisco_carboxylase(), n.rubp()), n.ki(n.rubisco_carboxylase(), n.pga()), n.ki(n.rubisco_carboxylase(), n.nadp()), n.ki(n.rubisco_carboxylase(), n.adp()), n.ki(n.rubisco_carboxylase(), n.pi())]
    )
    model.add_derived(
        "f_rubp",
        fn=_f_rubp,
        args=[n.rubp(), "Et", n.km(n.rubisco_carboxylase(), f"{n.rubp()}_extra")]
    )
    model.add_derived(
        n.o2(),
        fn=div,
        args=["p_o2", "Kh_o2"]
    )
    model.add_derived(
        "Ract_eq",
        fn=_Ract_eq,
        args=[n.co2(), n.pfd(), "alpha_ppfd_rub", "V0_ppfd_rub", "theta_ppfd_rub", "alpha_co2", "V0_co2", "theta_co2"]
    )
    model.add_derived(
        "I2,0",
        fn=_i20,
        args=[n.pfd(), "s"]
    )
    model.add_derived(
        "I1,0",
        fn=_i10,
        args=["I2,0", "Y(II)_LL", "Y(I)_LL"]
    )
    model.add_derived(
        "chi",
        fn=_chi,
        args=["f_cyc", "Y(II)_LL"]
    )
    model.add_derived(
        "I1",
        fn=_i1,
        args=["chi", "I1,0"]
    )
    model.add_derived(
        "f_cyc",
        fn=_f_cyc,
        args=["J_ATP", "J_NADPH", "v_ATP", "v_NADPH"]
    )
    model.add_derived(
        "I2",
        fn=_i2,
        args=["Y(II)_LL", "chi", "I2,0"]
    )
    model.add_derived(
        "Y(II)",
        fn=_y_ii,
        args=["Y(II)_LL", "v_ATP", "J_ATP", "v_NADPH", "J_NADPH", n.pfd(), "alpha_ppfd_Y(II)", "V0_ppfd_Y(II)", "theta_ppfd_Y(II)"]
    )
    model.add_derived(
        "J2",
        fn=_j2,
        args=["I2", "Y(II)"]
    )
    model.add_derived(
        "J1",
        fn=_j1,
        args=["J2", "f_cyc"]
    )
    model.add_derived(
        "f_pseudocyc",
        fn=_f_pseudocyc,
        args=["J_NADPH", n.o2(), "v_NADPH", "f_pseudocycNR"]
    )
    model.add_derived(
        "J_NADPH_steady",
        fn=_j_nadph_steady,
        args=["J1", "f_cyc", "f_pseudocyc"]
    )
    model.add_derived(
        "J_ATP_steady",
        fn=_j_atp_steady,
        args=["J2", "J1", "f_cyc", "fq", "f_ndh", "h"]
    )
    model.add_derived(
        "gs_steady",
        fn=_gs_steady,
        args=["tau0", "f_rubp", "chi_beta", "phi", "pi_e", "Kh", "Ds", "gs0"]
    )
    
    # Simple ODEs
    model.add_reaction(
        "Ract_rate", # Rubisco activation rate
        fn=ract_gs_time_dependance,
        args=["Ract", "Ract_eq", "tau_i", "tau_d"],
        stoichiometry={"Ract": 1}
    )
    model.add_reaction(
        "v_J_NADPH",
        fn=atp_nadph_time_dependance,
        args=["J_NADPH", "J_NADPH_steady", "Kj_NADPH"],
        stoichiometry={
            "J_NADPH": 1
        }
    )
    model.add_reaction(
        "v_J_ATP",
        fn=atp_nadph_time_dependance,
        args=["J_ATP", "J_ATP_steady", "Kj_ATP"],
        stoichiometry={
            "J_ATP": 1
        }
    )
    model.add_reaction(
        rxn := "v_gs",
        fn=ract_gs_time_dependance,
        args=["gs", "gs_steady", "Ki", "Kd"],
        stoichiometry={"gs": 1}
    )
    
    # Rates
    model.add_reaction(
        rxn := n.rubisco_carboxylase(),
        fn=_rubisco_carboxylation_bellasio,
        args=[
            n.rubp(),
            n.co2(),
            "Ract",
            n.km(rxn, n.co2()),
            n.o2(),
            n.km(rxn, n.o2()),
            n.vmax(rxn),
            "f_rubp",
            n.km(n.rubisco_carboxylase(), f"{n.rubp()}_extra")
        ],
        stoichiometry= {
            n.co2(): Derived(fn=neg_one_div, args=["V_m"]),
            n.rubp(): Derived(fn=neg_one_div, args=["V_m"]),
            n.pga(): Derived(fn=two_div, args=["V_m"])
        }
    )
    model.add_reaction(
        n.rubisco_oxygenase(),
        fn=_rubisco_oxygenase_bellasio,
        args=[n.co2(), n.o2(), "S_co", n.rubisco_carboxylase()],
        stoichiometry={
            n.rubp(): Derived(fn=neg_one_div, args=["V_m"]),
            n.pga(): Derived(fn=one_div, args=["V_m"]),
            n.atp(): Derived(fn=neg_one_div, args=["V_m"]),
            n.nadph(): Derived(fn=neg_half_div, args=["V_m"]),
        }
    )
    model.add_reaction(
        n.glycine_decarboxylase(),
        fn=value,
        args=[n.rubisco_oxygenase()],
        stoichiometry={
            n.co2(): Derived(fn=half_div, args=["V_m"]),
            n.pga(): Derived(fn=half_div, args=["V_m"]),
        }
    )
    model.add_reaction(
        rxn := n.r1p_kinase(), # RuP_phosp
        fn=_prkase,
        args=[n.atp(), n.rubp(), n.ru5p(), n.pga(), n.adp(), n.pi(), n.vmax(rxn), n.keq(rxn), n.km(rxn, n.atp()), n.ki(rxn, n.adp()), n.km(rxn, n.ru5p()), n.ki(rxn, n.pga()), n.ki(rxn, n.rubp()), n.ki(rxn, n.pi())],
        stoichiometry={
            n.rubp(): Derived(fn=one_div, args=["V_m"]),
            n.dhap(): Derived(fn=neg_fivethirds_div, args=["V_m"]),
            n.atp(): Derived(fn=neg_one_div, args=["V_m"]),
            n.ru5p(): Derived(fn=neg_one_div, args=["V_m"])
        }
    )
    model.add_reaction(
        rxn := "v_pgareduction", # PR
        fn=_v_pgareduction,
        args=[n.atp(), n.pga(), n.nadph(), n.adp(), n.vmax(rxn), n.km(rxn, n.atp()), n.km(rxn, n.pga()), n.km(rxn, n.nadph()), n.ki(rxn, n.adp())],
        stoichiometry={
            n.pga(): Derived(fn=neg_one_div, args=["V_m"]),
            n.dhap(): Derived(fn=one_div, args=["V_m"]),
            n.atp(): Derived(fn=neg_one_div, args=["V_m"]),
            n.nadph(): Derived(fn=neg_one_div, args=["V_m"])
        }
    )
    model.add_reaction(
        rxn := "v_carbohydrate_synthesis",
        fn=_v_carbohydrate_synthesis,
        args=[n.dhap(), n.pi(), n.adp(), n.vmax(rxn), "v_pgareduction", n.keq(rxn), n.km(rxn, n.dhap()), n.ki(rxn, n.adp())],
        stoichiometry={
            n.dhap(): Derived(fn=neg_one_div, args=["V_m"]),
            n.atp(): Derived(fn=neg_half_div, args=["V_m"]),
        }
    )
    model.add_reaction(
        rxn := "v_rpp",
        fn=_v_rpp,
        args=[n.dhap(), n.ru5p(), n.vmax(rxn), n.keq(rxn), n.km(rxn, n.dhap())],
        stoichiometry={
            n.ru5p(): Derived(fn=one_div, args=["V_m"])
        }
    )
    model.add_reaction(
        rxn := "v_co2_hydration", # CA
        fn=_v_co2_hydration,
        args=[n.co2(), n.hco3(), n.h(), n.vmax(rxn), n.keq(rxn), n.km(rxn, n.co2()), n.km(rxn, n.hco3())],
        stoichiometry={
            n.co2(): Derived(fn=neg_one_div, args=["V_m"]),
            "HCO3": Derived(fn=one_div, args=["V_m"])
        }
    )
    model.add_reaction(
        "v_RLight",
        fn=value,
        args=["RLight"],
        stoichiometry={
            n.co2(): Derived(fn=one_div, args=["V_m"]),
            n.pga(): Derived(fn=neg_onethirds_div, args=["V_m"])
        }
    )
    model.add_reaction(
        rxn := "v_NADPH",
        fn=_v_NADPH,
        args=[n.nadph(), n.nadp(), "J_NADPH", n.keq(rxn), n.km(rxn, n.nadp()), n.km(rxn, n.nadph())],
        stoichiometry={
            n.nadph(): Derived(fn=one_div, args=["V_m"])
        }
    )
    model.add_reaction(
        rxn := "v_ATP",
        fn=_v_atp,
        args=[n.atp(), n.adp(), n.pi(), "J_ATP", n.keq(rxn), n.km(rxn, n.adp()), n.km(rxn, n.pi()), n.km(rxn, n.atp())],
        stoichiometry={
            n.atp(): Derived(fn=one_div, args=["V_m"])
        }
    )
    # CO2 Dissolution
    model.add_reaction(
        "CO2 dissolution",
        fn=_co2_diss,
        args=["Ci", n.co2(), "gm", "Kh_co2"],
        stoichiometry={
            "Ci": -1,
            n.co2(): Derived(fn=one_div, args=["V_m"])
        }
    )
    # co2 stomatal diffusion
    model.add_reaction(
        "CO2 stomatal diffusion",
        fn=_stom_diff,
        args=["Ci", "gs", "Ca"],
        stoichiometry={
            "Ci": 1
        }
    )
    
    return model
