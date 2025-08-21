from mxlpy import Model, Derived
import mxlbricks.names as n
from mxlbricks.fns import mul
import numpy as np

def _v_PSII_recomb(Dy, pH_lumen, QAm, k_recomb):
    delta_G_recomb = Dy + 0.06 * (7.0 - pH_lumen)
    return k_recomb * QAm + 10 ** (delta_G_recomb / 0.06)

def _PSII_charge_separations(PSII_antenna_size, light_per_L, Phi2):
    return PSII_antenna_size * light_per_L * Phi2

def get_li2021(
    *,
    chl_lumen: str = "_lumen",
    chl_stroma: str = "_stroma",
) -> Model:
    
    model = Model()
    
    model.add_variables({
        "QA": 1, # QA_content
        "QA_red": 0, # Qm_content
        n.pq_ox(): 7, # PQ_content
        n.pq_red(): 0, # PQH2_content
        n.h(): 0.0, # Hin
        n.ph(chl_lumen): 7.8, # pHlumen
        "Dy": 0.0, # Dy
        "pmf": 0.0, # pmf
        "DeltaGatp": 30.0 + 2.44 * np.log(1/0.0015), # DeltaGatp probably parameter
        n.pottassium(chl_lumen): 0.1, # Klumen
        n.pottassium(chl_stroma): 0.1, # Kstroma # Probably parameter
        n.atp(): 0, # ATP_made
        n.pc_ox(): 0, # PC_ox
        n.pc_red(): 2, # PC_red
        n.a0(): 0.0, # P700_ox
        n.a1(): 0.667, # P700_red
        n.zx(): 0.0, # Z
        n.vx(): 1, # V
        "NPQ": 0, # NPQ
        "singletO2": 0, # singletO2
        "Phi2": 0.83, # Phi2
        "LEF": 0, # LEF
        n.fd_red(): 0, # Fd_red
        n.fd_ox(): 1, # Fd_ox
        n.atp("_pool"): 4.15, # ATP_pool
        n.adp("_pool"): 4.15, # ADP_pool
        n.nadph("_pool"): 1.5, # NADPH_pool
        n.nadp("_pool"): 3.5, # NADP_pool
        n.chloride(chl_lumen): 0.04, # Cl_lumen
        n.chloride(chl_stroma): 0.04, # Cl_stroma
        n.h(chl_stroma): 0, # Hstroma
        n.ph(chl_stroma): 7.8, # phstroma
    })
    
    model.add_parameters({
        "k_recomb": 0.33,
        "triplet_yield": 0.45,
        "triplet_to_singletO2_yield": 1
    })
    
    # Rates
    model.add_reaction(
        "v_PSII_recomb",
        fn=_v_PSII_recomb,
        args=["Dy", "pH_lumen", "QAm", "k_recomb"],
        stoichiometry={
            "singletO2": Derived(fn=mul, args=["triplet_yield", "triplet_to_singletO2_yield"])
        }
    )

    model.add_reaction(
        name="PSII_charge_separations",
        fn=_PSII_charge_separations,
        args=["PSII_antenna_size", "light_per_L", "Phi2"],
        stoichiometry={
            "QAm": 1,
            "QA": -1
        }
    )
    
    return model

print(get_li2021())