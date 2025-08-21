from scipy.integrate import solve_ivp
import numpy as np

def _v_PSII_recombination(Dy, QAm, pH_lumen, k_recomb):
    delta_delta_g_recomb= Dy + 0.06 * (7.0 - pH_lumen)
    return k_recomb * QAm * 10**(delta_delta_g_recomb / 0.06)

def _v_PSII_charge_separations(antenna_size, light_per_L, Phi2):
    return antenna_size * light_per_L * Phi2

def _v_b6f(pH_lumen, PQH2, PQ, PC_ox, PC_red, pKreg, b6f_content, Em7_PC, Em7_PQH2, pmf, max_b6f):
    pHmod=(1 - (1 / (10 ** (pH_lumen - pKreg) + 1)))
    b6f_deprot=pHmod*b6f_content

    Em_PC=Em7_PC
    Em_PQH2= Em7_PQH2 - 0.06*(pH_lumen-7.0)

    Keq_b6f = 10**((Em_PC - Em_PQH2 - pmf)/.06)
    k_b6f=b6f_deprot * max_b6f

    k_b6f_reverse = k_b6f / Keq_b6f
    #print('Keq for PQH2 to PC + pmf is: ' + str(Keq_b6f))
    f_PQH2=PQH2/(PQH2+PQ) #want to keep the rates in terms of fraction of PQHs, not total number
    f_PQ=1-f_PQH2
    return f_PQH2 * PC_ox * k_b6f - f_PQ * PC_red * k_b6f_reverse 

def _v_NDH(Fd_red, PQ, Fd_ox, PQH2, pH_stroma, Em7_PQH2, Em_Fd, k_NDH, pmf):
    Em_PQH2 = Em7_PQH2 - 0.06*(pH_stroma - 7.0)
    deltaEm = Em_PQH2 - Em_Fd
    Keq_NDH = 10**((deltaEm - pmf*2)/0.06)
    k_NDH_reverse = k_NDH / Keq_NDH
    return k_NDH * Fd_red * PQ - k_NDH_reverse * Fd_ox * PQH2

def _v_PGR(Fd_red, PQ, PQH2, PGR_vmax):
    return PGR_vmax * (Fd_red**4/(Fd_red**4+0.1**4))*PQ/(PQ+PQH2)
    
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
    return k_CBC*(1.0-np.exp(-t/600))*(np.log(NADPH_pool/NADP_pool)-np.log(1.25))/(np.log(3.5/1.25))

def _v_KEA(QAm, pH_lumen, K_lumen, H_lumen, H_stroma, K_stroma, k_KEA):
    qL = 1 - QAm
    qL_act = qL**3/(qL**3+0.15**3)
    pH_act =1/(10**(1*(pH_lumen-6.0))+1)
    f_KEA_act = qL_act * pH_act
    return k_KEA * (H_lumen * K_stroma -  H_stroma * K_lumen) * f_KEA_act

def _v_K_channel(K_lumen, Dy, K_stroma, perm_K):
    K_deltaG = -0.06*np.log10(K_stroma/K_lumen) + Dy
    return perm_K * K_deltaG*(K_lumen+K_stroma)/2

def _v_VCCN1(Cl_lumen, Cl_stroma, driving_force_Cl, k_VCCN1):
    relative_Cl_flux = 332*(driving_force_Cl**3) + 30.8*(driving_force_Cl**2) + 3.6*driving_force_Cl
    return k_VCCN1 * relative_Cl_flux * (Cl_stroma + Cl_lumen)/2

def _v_CLCE(Cl_lumen, Cl_stroma, H_lumen, H_stroma, driving_force_Cl, pmf, k_CLCE):
    return k_CLCE*(driving_force_Cl*2+pmf)*(Cl_stroma + Cl_lumen)*(H_lumen+H_stroma)/4

def _v_leak(H_lumen, pmf, k_leak):
    return pmf*k_leak*H_lumen

def _v_pmf_protons_activity(t, pmf, n, ATP_synthase_max_turnover, light_per_L):
    x = t/165
    actvt = 0.2 + 0.8*(x**4/(x**4 + 1))
    v_proton_active = 1 - (1 / (10 ** ((pmf - 0.132)*1.5/0.06) + 1))#reduced ATP synthase
    v_proton_inert = 1-(1 / (10 ** ((pmf - 0.204)*1.5/0.06) + 1))#oxidized ATP synthase
    
    v_active = actvt * v_proton_active * n * ATP_synthase_max_turnover
    v_inert = (1-actvt) * v_proton_inert * n * ATP_synthase_max_turnover
    
    v_proton_ATP = v_active + v_inert
    
    if light_per_L > 0:
        return v_proton_ATP
    else:
        return 0
    
def _v_ZE(Z, kZE):
    return Z * kZE

def _v_VDE(V, pH_lumen, VDE_Hill, pKvde, VDE_max_turnover_number):
    pHmod= 1 / (10 ** (VDE_Hill*(pH_lumen - pKvde)) + 1)
    return V * VDE_max_turnover_number * pHmod

def calc_phi2(NPQ, QA):
    return 1 / (1 + ( 1 + NPQ)/ (4.88 * QA))

def moeity(total, x):
    return total - x

def calc_PsbS_Protonation(pH_lumen, pKPsbS):
    return 1 / (10 ** (3*(pH_lumen - pKPsbS)) + 1)

def calc_h(pH):
    return 10**(-1*pH)

def odes(t, y, PAR):
    
    QAm, PQH2, pH_lumen, Dy, K_lumen, PC_ox, P700_ox, Z, singletO2, Fd_red, NADPH_pool, Cl_lumen, Cl_stroma = y
    
    # Parameters
    k_recomb = 0.33
    triplet_yield = 0.45
    triplet_to_singletO2_yield = 1
    PSII_antenna_size = 0.5
    QA_total = 1
    b6f_content=0.433
    pKreg = 6.2
    Em7_PC = 0.37
    Em7_PQH2 = 0.11
    max_b6f = 6.2
    PQ_total = 7
    P700_total = 0.667
    PC_total = 2
    Fd_total = 1
    pKPsbS = 6.2
    max_NPQ = 3
    pH_stroma = 7.8
    Em_Fd = -0.42
    k_NDH = 1000
    PGR_vmax = 0
    PSI_antenna_size = 0.5
    NA_total = 5
    kQA = 1000
    Keq_QA_PQ = 200
    k_PC_to_P700 = 5000
    k_Fd_to_NADP = 1000
    k_CBC = 60
    K_stroma = 0.1
    k_KEA = 2500000
    perm_K = 150
    lumen_protons_per_turnover = 0.000587
    k_VCCN1 = 12
    k_CLCE = 800000
    k_leak = 3*10**7
    n = 14/3
    ATP_synthase_max_turnover = 200
    buffering_capacity = 0.014
    Volts_per_charge = 0.047
    kZE = 0.004
    VDE_Hill = 4
    pKvde = 5.65
    VDE_max_turnover_number = 0.08
    X_total = 1
    
    # Derived Constants
    light_per_L = 0.84 * PAR / 0.7
    driving_force_Cl = 0.06* np.log10(Cl_stroma/Cl_lumen) + Dy
    
    # Derived Past
    PsbS_H = calc_PsbS_Protonation(pKPsbS, pH_lumen)
    NPQ = 0.4 * max_NPQ * PsbS_H * Z + 0.5 * max_NPQ * PsbS_H + 0.1 * max_NPQ * Z
    QA = moeity(QA_total, QAm)
    Phi2 = calc_phi2(NPQ, QA)
    PQ = moeity(PQ_total, PQH2)
    P700_red = moeity(P700_total, P700_ox)
    PC_red = moeity(PC_total, PC_ox)
    Fd_ox = moeity(Fd_total, Fd_red)
    NADP_pool = moeity(NA_total, NADPH_pool)
    H_lumen = calc_h(pH_lumen)
    H_stroma = calc_h(pH_stroma)
    V = moeity(X_total, Z)
    pmf = Dy + 0.06*(pH_stroma-pH_lumen)
    
    #Rates
    v_PSII_recombination = _v_PSII_recombination(Dy, QAm, pH_lumen, k_recomb)
    v_PSII_charge_separations = _v_PSII_charge_separations(PSII_antenna_size, light_per_L, Phi2)
    v_PQ_reduction_QA = _v_PQ_reduction_QA(QAm, PQ, kQA) # Replaced loose quation
    v_PQH2_oxidation_QA = _v_PQH2_oxidation_QA(PQH2, QA, kQA, Keq_QA_PQ) # Replaced loose quation
    v_b6f = _v_b6f(pH_lumen, PQH2, PQ, PC_ox, PC_red, pKreg, b6f_content, Em7_PC, Em7_PQH2, pmf, max_b6f)
    v_NDH = _v_NDH(Fd_red, PQ, Fd_ox, PQH2, pH_stroma, Em7_PQH2, Em_Fd, k_NDH, pmf)
    v_PGR = _v_PGR(Fd_red, PQ, PQH2, PGR_vmax)
    v_PSI_charge_separation = _v_PSI_charge_separation(Fd_ox, P700_red, PSI_antenna_size, light_per_L)
    v_PC_oxidiation_P700 = _v_PC_oxidation_P700(PC_red, P700_ox, k_PC_to_P700) # Replaced loose quation
    v_LEF = _v_LEF(Fd_red, NADP_pool, k_Fd_to_NADP) # Replaced dLEF
    v_Mehler = _v_Mehler(Fd_red, Fd_ox)
    v_CBB_NADPH = _v_CBB_NADPH(NADPH_pool, NADP_pool, t, k_CBC)
    v_KEA = _v_KEA(QAm, pH_lumen, K_lumen, H_lumen, H_stroma, K_stroma, k_KEA)
    v_K_channel = _v_K_channel(K_lumen, Dy, K_stroma, perm_K)
    v_VCCN1 = _v_VCCN1(Cl_lumen, Cl_stroma, driving_force_Cl, k_VCCN1)
    v_CLCE =  _v_CLCE(Cl_lumen, Cl_stroma, H_lumen, H_stroma, driving_force_Cl, pmf, k_CLCE)
    v_leak = _v_leak(H_lumen, pmf, k_leak)
    v_pmf_protons_activity = _v_pmf_protons_activity(t, pmf, n, ATP_synthase_max_turnover, light_per_L)
    v_ZE = _v_ZE(Z, kZE)
    v_VDE = _v_VDE(V, pH_lumen, VDE_Hill, pKvde, VDE_max_turnover_number)
    
    # ODEs
    dsingletO2 = (triplet_yield * triplet_to_singletO2_yield) * v_PSII_recombination
    dQAm = v_PSII_charge_separations  + v_PQH2_oxidation_QA - v_PQ_reduction_QA - v_PSII_recombination
    dPQH2 = 0.5 * v_PQ_reduction_QA + 0.5 * v_NDH + 0.5 * v_PGR - 0.5 * v_b6f - 0.5 * v_PQH2_oxidation_QA
    dP700_ox = v_PSI_charge_separation - v_PC_oxidiation_P700
    dPC_ox = v_PC_oxidiation_P700 - v_b6f
    dFd_red= v_PSI_charge_separation - v_LEF - v_NDH - v_PGR - v_Mehler
    dNADPH_pool= 0.5 * v_LEF - v_CBB_NADPH
    dK_lumen = (v_KEA - v_K_channel) * lumen_protons_per_turnover
    dCl_lumen = (v_VCCN1 + 2 * v_CLCE) * lumen_protons_per_turnover
    dCl_stroma = -0.1 * (v_VCCN1 + 2 * v_CLCE) * lumen_protons_per_turnover
    dpH_lumen=-1 * ((v_PSII_charge_separations - v_PSII_recombination + v_b6f*2 + 2 * v_NDH - v_pmf_protons_activity + v_leak - v_KEA - v_CLCE)*lumen_protons_per_turnover) / buffering_capacity
    dDy= (v_PSII_charge_separations - v_PSII_recombination + v_PSI_charge_separation + v_b6f + 2 * v_NDH - v_K_channel - v_pmf_protons_activity - v_leak - v_VCCN1 - 3 * v_CLCE) * Volts_per_charge
    dZ = v_VDE - v_ZE
    
    
    # Derived Present
    
    
    return [dQAm, dPQH2, dpH_lumen, dDy, dK_lumen, dPC_ox, dP700_ox, dZ, dsingletO2, dFd_red, dNADPH_pool, dCl_lumen, dCl_stroma]
    
    