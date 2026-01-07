from mxlpy import Model, Variable, InitialAssignment, Parameter, units, Derived
import mxlbricks.names as n
from added_fns import half, two_div, neg_one_div, neg_mul
from added_names import total_psii_rc, quencher_active, quencher_inactive, total_quencher
from mxlbricks.fns import moiety_2, moiety_1, mass_action_2s, mass_action_1s, one_div, neg_div, neg, mul, value, twice
import numpy as np

def _v3(A2: float, pq_ox: float, A0: float, pq_red: float, k3p: float, k3m: float) -> float:
    # TODO: switched pqs than it was in paper, check if correct
    return k3p * A2 * pq_red - k3m * A0 * pq_ox

def _v5(ATP: float, A_tot: float, k5: float, T: float, R: float, DeltaG_ATP: float, H_st: float, H_lu: float, Pi: float) -> float:
    K_eq = Pi * np.exp(-(DeltaG_ATP) / (R * T)) * (H_st / H_lu) ** (14/3)
    return k5 * (A_tot - ATP * (1 + 1 / K_eq))

def _v6(Q_inactive: float, H_lu: float, n: float, K_Q: float, k6: float) -> float:
    return k6 * Q_inactive * H_lu**n / (H_lu**n + K_Q**n)

def _v8(H_lu: float, H_st: float, k8: float) -> float:
    return k8 * (H_lu - H_st)

def get_ebenhoeh2011() -> Model:
    """
    'A minimal mathematical model of nonphotochemical quenching of chlorophyll fluorescence' by Ebenhoeh et al., 2011.
    
    Biosystems 2011 (https://doi.org/10.1016/j.biosystems.2010.10.011.)
    """
    
    model = Model()
    
    model.add_variables({
        n.a0(): Variable(initial_value=InitialAssignment(fn=half, args=[total_psii_rc()]), unit=units.mmol_mol_chl), #A1 in paper
        n.a1(): Variable(initial_value=InitialAssignment(fn=half, args=[total_psii_rc()]), unit=units.mmol_mol_chl),  #A2 in paper,
        n.pq_ox(): Variable(initial_value=0, unit=units.mmol_mol_chl),  #P in paper
        n.h("_lumen"): Variable(initial_value=6.34e-5, unit=units.mmol_mol_chl),  #H in paper
        quencher_active(): Variable(initial_value=0, unit=units.mmol_mol_chl), #N in paper
        n.atp("_stroma"): Variable(initial_value=0, unit=units.mmol_mol_chl), #T in paper
    })
    
    model.add_parameters({
        total_psii_rc(): Parameter(value=2.5, unit=units.mmol_mol_chl, source="https://doi.org/10.1104/pp.104.052324"),
        n.total_pq(): Parameter(value=17.5, unit=units.mmol_mol_chl, source="https://doi.org/10.1021/bi011650y"),
        n.total_adenosines(): Parameter(value=32, unit=units.mmol_mol_chl, source="https://doi.org/10.1104/pp.88.4.1461"),
        total_quencher(): Parameter(value=1, unit=units.mmol_mol_chl),
        n.pfd(): Parameter(value=1000, unit=units.mumol / units.sqm / units.second),  #Photon flux density # TODO: Check unit
        n.convf(): Parameter(value=4/3),  #Conversion factor
        "bH": Parameter(value=0.01, unit=units.dimensionless, source="https://doi.org/10.1007/s11120-006-9109-1"),  #Buffering capacity
        "k2": Parameter(value=3.4e6, unit=units.per_second, source="https://doi.org/10.1016/0005-2728(79)90063-X"),  #s^-1
        "k3p": Parameter(value=1.56e5, unit=units.mmol_mol_chl* units.per_second, source="https://doi.org/10.1016/0005-2728(76)90038-4"),  # 
        "k3m": Parameter(value=3.12e4, unit=units.mmol_mol_chl * units.per_second, source="https://europepmc.org/article/med/12500567"),  #
        "k4": Parameter(value=50, unit=units.per_second),  #s^-1
        "k5": Parameter(value=80, unit=units.per_second, source="https://doi.org/10.1016/0014-5793(96)00246-3"),  #s^-1
        "T": Parameter(value=298, unit=units.kelvin),  #Temperature
        "R": Parameter(value=0.0083, unit=units.joule * 1000 / (units.mol * units.kelvin)),  #Gas constant
        "DeltaG_ATP": Parameter(value=30.6, unit=units.joule * 1000 / units.mol, source="https://doi.org/10.1016/0005-2728(72)90116-8"),  #Gibbs free energy of ATP hydrolysis
        n.h("_stroma"): Parameter(value=6.34e-5, unit=units.mmol_mol_chl),  #H in stroma
        n.pi(): Parameter(value=0.01, unit=units.mmol / units.liter, source="https://doi.org/10.1104/pp.88.4.1461"),  #Inorganic phosphate concentration # TODO: Check unit
        "HPR": Parameter(value=14/3, unit=units.dimensionless),
        "n": Parameter(value=5, unit=units.dimensionless, source="https://doi.org/10.1104/pp.101.1.651"),  #Hill coefficient
        "K_Q": Parameter(value=0.004, unit=units.mmol_mol_chl),
        "k6": Parameter(value=0.05, unit=units.per_second, source="https://doi.org/10.1104/pp.101.1.65"),
        "k7": Parameter(value=0.004, unit=units.per_second, source="https://doi.org/10.1104/pp.101.1.65"),
        "k8": Parameter(value=10, unit=units.per_second),
        "k9": Parameter(value=20, unit=units.per_second),
    })
    
    # Derived variables
    model.add_derived(
        n.a2(),
        fn=moiety_2,
        args=[n.a0(), n.a1(), total_psii_rc()],
        unit=units.mmol_mol_chl
    )
    
    model.add_derived(
        name=n.pq_red(),
        fn=moiety_1,
        args=[n.pq_ox(), n.total_pq()],
    )
    
    model.add_derived(
        name=n.adp("_stroma"),
        fn=moiety_1,
        args=[n.atp("_stroma"), n.total_adenosines()],
        unit=units.mmol_mol_chl
    )
    
    model.add_derived(
        name=quencher_inactive(),
        fn=moiety_1,
        args=[quencher_active(), total_quencher()],
        unit=units.mmol_mol_chl
    )
    
    model.add_derived(
        name="k1",
        fn=mul,
        args=[n.pfd(), n.convf()],
    )
    
    #Rates
    
    model.add_reaction(
        name="v1",
        fn=mass_action_2s,
        args=[n.a0(), quencher_inactive(), "k1"],
        stoichiometry={n.a0(): -1, n.a1(): 1},
    )
    
    model.add_reaction(
        name="v2",
        fn=mass_action_1s,
        args=[n.a1(), "k2"],
        stoichiometry={n.a1(): -1, n.h("_lumen"): Derived(fn=twice, args=["bH"])},
    )
    
    model.add_reaction(
        name="v3",
        fn=_v3,
        args=[n.a2(), n.pq_ox(), n.a0(), n.pq_red(), "k3p", "k3m"],
        stoichiometry={n.a0(): 1, n.pq_ox(): 1},
    )
    
    model.add_reaction(
        name="v4",
        fn=mass_action_1s,
        args=[n.pq_ox(), "k4"], # TODO: changed to pq_ox from pq_red, check if correct
        stoichiometry={n.pq_ox(): -1, n.h("_lumen"): Derived(fn=value, args=["bH"])},
    )
    
    model.add_reaction(
        name="v5",
        fn=_v5,
        args=[n.atp("_stroma"), n.total_adenosines(), "k5", "T", "R", "DeltaG_ATP", n.h("_stroma"), n.h("_lumen"), n.pi()],
        stoichiometry={n.h("_lumen"): Derived(fn=neg_mul, args=["HPR", "bH"]), n.atp("_stroma"): 1},
    )
    
    model.add_reaction(
        name="v6",
        fn=_v6,
        args=[quencher_inactive(), n.h("_lumen"), "n", "K_Q", "k6"],
        stoichiometry={quencher_active(): 1},
    )
    
    model.add_reaction(
        name="v7",
        fn=mass_action_1s,
        args=[quencher_active(), "k7"],
        stoichiometry={quencher_active(): -1},
    )
    
    model.add_reaction(
        name="v8",
        fn=_v8,
        args=[n.h("_lumen"), n.h("_stroma"), "k8"],
        stoichiometry={n.h("_lumen"): Derived(fn=neg, args=["bH"])},
    )
    
    model.add_reaction(
        name="v9",
        fn=mass_action_1s,
        args=[n.atp("_stroma"), "k9"],
        stoichiometry={n.atp("_stroma"): -1},
    )
    
    return model