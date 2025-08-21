def f(y, t, pKreg, max_PSII, kQA, max_b6f, lumen_protons_per_turnover, PAR, ATP_synthase_max_turnover, 
    PSII_antenna_size, Volts_per_charge, perm_K, n, Em7_PQH2, Em7_PC,Em_Fd, PSI_antenna_size, 
    buffering_capacity, VDE_max_turnover_number, pKvde, VDE_Hill, kZE, pKPsbS, max_NPQ, k_recomb, k_PC_to_P700, 
    triplet_yield, triplet_to_singletO2_yield, fraction_pH_effect, k_Fd_to_NADP, k_CBC, k_KEA, k_VCCN1, k_CLCE, k_NDH): 

    #The following are holders for paramters for testing internal functions of f
    light_per_L=0.84 * PAR/0.7
    #we got 4.1 nmol Chl per 19.6 mm^2 leaf disc, which translate into 210 umol Chl/m2
    #210 umol Chl/m2, PSII/300 Chl ==> 0.7 umol PSII/m2, ==>(PAR/0.7) photons/PSII
    ###So the light_per_L means the photons/PSII that hit all the thylakoid membranes, and absorbed by the leaf
    #the 0.84 is the fraction of light a leaf typically absorbs

    
    QA, QAm, PQ, PQH2, Hin, pHlumen, Dy, pmf, deltaGatp, Klumen, Kstroma, ATP_made,\
    PC_ox, PC_red, P700_ox, P700_red, Z, V, NPQ, singletO2, Phi2, LEF, Fd_ox, Fd_red,\
    ATP_pool, ADP_pool, NADPH_pool, NADP_pool,Cl_lumen, Cl_stroma, Hstroma, pHstroma = y
    
    print(y)
    
    PSII_recombination_v=recombination_with_pH_effects(k_recomb, QAm, Dy, pHlumen, fraction_pH_effect)
        
    dsingletO2=PSII_recombination_v*triplet_yield*triplet_to_singletO2_yield

    #calculate pmf from Dy and deltapH 
    pmf=Dy + 0.06*(pHstroma-pHlumen)

    #***************************************************************************************
    #PSII reations
    #****************************************************************************************
    #first, calculate Phi2
    Phi2=Calc_Phi2(QA, NPQ) #I use the current' value of NPQ. I then calculate the difference below 

    #calculate the number of charge separations in PSII per second
    PSII_charge_separations=PSII_antenna_size*light_per_L * Phi2
    
    #The equilibrium constant for sharing electrons between QA and the PQ pool
    #This parameter will be placed in the constants set in next revision
    
    Keq_QA_PQ=200
    
    #calculate the changes in QA redox state based on the number of charge separations and equilibration with 
    #the PQ pool
    dQAm = PSII_charge_separations  + PQH2*QA*kQA/Keq_QA_PQ  - QAm * PQ * kQA - PSII_recombination_v
    dQA = -1*dQAm

    #***************************************************************************************
    #PQ pool and the cyt b6f complex
    #***************************************************************************************

    #vb6f = k_b6f(b6f_max_turnover_number, b6f_content, pHlumen, pKreg, PQH2)

    #b6f_content describes the relative (to standaard PSII) content of b6f 
    #This parameter will be placed in the constants set in next revision
    b6f_content=0.433 #Journal of Experimental Botany, Vol. 65, No. 8, pp. 1955–1972, 2014
    #doi:10.1093/jxb/eru090 Advance Access publication 12 March, 2014
    #Mathias Pribil1, Mathias Labs1 and Dario Leister1,2,* Structure and dynamics of thylakoids in land plantsJournal of Experimental Botany, Vol. 65, No. 8, pp. 1955–1972, 2014
   
    #calc_v_b6f return the rate of electron flow through the b6f complex
    v_b6f=calc_v_b6f(max_b6f, b6f_content, pHlumen, pKreg, PQ, PQH2, PC_ox, PC_red, Em7_PC, Em7_PQH2, pmf)
    
    v_NDH = calc_v_NDH(Em_Fd, Em7_PQH2, pHstroma, pmf, k_NDH, Fd_red, Fd_ox, PQ, PQH2)
    d_Hlumen_NDH = v_NDH*2 #change in lumen protons
    d_charge_NDH = d_Hlumen_NDH # change in charges across the membrane
    #d_Hstroma_NDH = v_NDH*3 # change in stroma protons
    
    ##PGR regulation, attempted
    PGR_vmax = 0#It seems this function does not impact the kinetics much.
    v_PGR = calc_v_PGR(PGR_vmax, Fd_red, PQ, PQH2)

    #calculate the change in PQH2 redox state considering the following:
    #PQ + QAm --> PQH2 + QA ; PQH2 + b6f --> PQ    
    PSI_charge_separations= P700_red * light_per_L * PSI_antenna_size * Fd_ox
    #aleternatively,
    #PSI_charge_separations = P700_red*light_per_L*PSI_antenna_size*FB/(FB+FB_minus)
    #PSI_to_Fd = FB_minus*Fd_ox*k_FB_Fd
    #d_FB_minus = PSI_charge_separations-PSI_to_Fd
    #d_FB = -d_FB_minus
    

    dPQH2 = (QAm * PQ * kQA + v_NDH + v_PGR - v_b6f - PQH2*QA*kQA/Keq_QA_PQ)*0.5 
    dPQ = -1*dPQH2

    #***************************************************************************************
    #PSI and PC reactions:
    #***************************************************************************************

    #Calculate the changes in PSI redox state. The current form is greatly simplified, 
    #but does consider the need for oxidized Fd.
    #At this point, I assumed that the total Fd pool is unity
    
    
    #P700 reactions
    d_P700_ox = PSI_charge_separations - PC_red * k_PC_to_P700 * P700_ox
    d_P700_red=-1*d_P700_ox
    
    #PC reactions:
    d_PC_ox = PC_red * k_PC_to_P700 * P700_ox - v_b6f
    d_PC_red = -1*d_PC_ox
    
    #Mehler reaction, V_me = kme * [O2]*Fd_red/(Fd_red+Fd_ox), Hui Lyu and Dusan Lazar modeling...
    V_me = 4*0.000265*Fd_red/(Fd_red+Fd_ox)
    dFd_red=PSI_charge_separations - k_Fd_to_NADP*Fd_red*NADP_pool - v_NDH - v_PGR -V_me
    dFd_ox=-1*dFd_red
    #alternatively,
    #dFd_red = PSI_to_Fd - k_Fd_to_NADP*Fd_red*NADP_pool - v_NDH-V_me
    
    #***************************************************************************************
    # ATP synthase reactions:
    #***************************************************************************************
    #However, one may also consider that there is a maximal (saturating turover rate 
    #(saturation point), as shown by Junesch and Grabber (1991)
    #http://dx.doi.org/10.1016/0014-5793(91)81447-G
    #    def Vproton(ATP_synthase_max_turnover, n, pmf, pmf_act):
    #    return (ATP_synthase_max_turnover*n*(1 - (1 / (10 ** ((pmf - pmf_act)/.06) + 1))))
    #vHplus=Vproton(ATP_synthase_max_turnover, n, pmf, pmf_act)
    
    #ATP_synthase_driving_force=pmf-(deltaGatp/n) #this is positive if pmf is sufficient to drive 
    #reaction forward, assuming ATP synthase activity is time dependent, derived from gH+ data
    # data courtesy from Geoff and Dave
    #Pi = 0.0025 - ATP_pool/7000
    #pmf_act = calc_pmf_act(ATP_pool, ADP_pool, Pi)
    Hlumen = 10**(-1*pHlumen)
    Hstroma = 10**(-1*pHstroma)
    
    activity = ATP_synthase_actvt(t)    
    d_protons_to_ATP = Vproton_pmf_actvt(pmf, activity, ATP_synthase_max_turnover, n)
    d_H_ATP_or_passive = V_H_light(light_per_L, d_protons_to_ATP, pmf, Hlumen)                              
    #d_protons_to_ATP_red = Vproton(ATP_synthase_max_turnover, n, pmf, pmf_act_red)*Patp_red
    #d_protons_to_ATP_ox = Vproton(ATP_synthase_max_turnover, n, pmf, pmf_act_ox)*Patp_ox
    #d_protons_to_ATP = d_protons_to_ATP_red + d_protons_to_ATP_ox
        
    d_ATP_made=d_protons_to_ATP/n                                        
    #The CBC is either limited by Phi2 or by the activation kinetics, take the minimum
    #NADPH_phi_2 = (PSII_charge_separations - PSII_recombination_v)*0.5

    NADPH_CBC = k_CBC*(1.0-np.exp(-t/600))*(np.log(NADPH_pool/NADP_pool)-np.log(1.25))/(np.log(3.5/1.25))#calc_CBC_NADPH(k_CBC, t, d_ATP_made)
    #this number in "np.exp(-t/600)" is important, which impacts the shape of the curves
    dNADPH_pool=0.5 * k_Fd_to_NADP*NADP_pool*Fd_red - NADPH_CBC
    dNADP_pool=-1*dNADPH_pool
    
    dLEF=k_Fd_to_NADP*NADP_pool*Fd_red
    
    d_ATP_consumed = d_ATP_made#NADPH_CBC*5/3 + (ATP_pool/(ADP_pool+ATP_pool)-0.5)*1.2#ATP_pool*(ATP_pool/ADP_pool-1)
    #***************************************************************************************
    #Proton input (from PSII, b6f and PSI) and output (ATP synthase) reactions :
    #***************************************************************************************
    #calculate the contributions to lumen protons from PSII, assuming a average of 
    #one released per S-state transition. In reality, the pattern is not 1:1:1:1, 
    #but for simplicity, I am assuming that the S-states are scrambled under our 
    #illumination conditions. This is described in more detail in the manuscript.
    
    d_protons_from_PSII = PSII_charge_separations - PSII_recombination_v

    #calculate the contributions to Dy from PSII
    charges_from_PSII = PSII_charge_separations - PSII_recombination_v
    
    #calculate the contributions to lumen protons from b6f complex
    #assuming the Q-cycle is engaged, asn thus
    #two protons are released into lumen per electron from
    #PQH2 to PC
    """
    C.A. Sacksteder, A. Kanazawa, M.E. Jacoby, D.M. Kramer (2000) The proton to electron 
    stoichiometry of steady-state photosynthesis in living plants: A proton-pumping Q-cycle 
    is continuously engaged. Proc Natl Acad Sci U S A 97, 14283-14288.

    """
    d_protons_from_b6f = v_b6f*2 #two protons per electron transferred from PQH2 to PC

    #calculate the contributions to Dy from Q-cycle turnover
    #one charge through the low potential b chain per
    #PQH2 oxidized
    charges_from_b6f = v_b6f
     
    #add up the changes in protons delivered to lumen
    #note: net_protons_in is the total number of protons input into the lumen, including both free and bound.
    net_protons_in = d_protons_from_PSII + d_protons_from_b6f + d_Hlumen_NDH - d_H_ATP_or_passive
    #net_protons_stroma = d_protons_to_ATP - v_b6f - d_Hstroma_NDH - QAm * PQ * kQA + PQH2*QA*kQA/Keq_QA_PQ  - dNADPH_pool - d_ATP_made
    #each ATP synthesis consumes one proton

    #see appendix for explanation
    
    #K_deltaG=0.06*np.log10(Kstroma/Klumen) - Dy
    
    #the KEA reaction looks like this:
    # H+(lumen) + K+(stroma) <-- --> H+(stroma) + K+(lumen)
    #and the reaction is electroneutral, 
    #so the forward reaction will depend on DpH and DK+ as:
    
    f_actvt = KEA_reg(pHlumen, QAm)
    v_KEA = k_KEA*(Hlumen*Kstroma -  Hstroma*Klumen)*f_actvt#/(10**(2*(pHlumen-6.5))+1)
    
    #Pck = 1/(1+np.exp(39.5*0.66*(-0.003-Dy)))#probability of v_K_channel open
    K_deltaG=-0.06*np.log10(Kstroma/Klumen) + Dy
    v_K_channel = perm_K * K_deltaG*(Klumen+Kstroma)/2
    
   
    #v_K_channel = Pck*perm_K * Dy * 39.5*(Klumen- Kstroma*np.exp(-39.5*Dy))/(1-np.exp(-39.5*Dy))#eq regular
    #v_K_channel = Pck*perm_K * (Klumen*np.exp(39.5*Dy)- Kstroma*np.exp(-39.5*Dy))#eq Hui Lyu
    #Adjusted from Hui Lyu and Dusan Lazar Journal of Theoretical Biology 413 (2017) 11-23, 39.5 = F/RT
    #It seems the flux of K+ is behave similar between Kramer and Lazar simulations.
    #Now the equation considers the  Goldman–Hodgkin–Katz flux equation
    #Hille, Bertil (2001) Ion channels of excitable membranes, 3rd ed.,p. 445, ISBN 978-0-87893-321-1
    
    #Next, use this to calculate a flux, which depends
    #on the permeability of the thylakoid to K+, perm_K:
    net_Klumen =  v_KEA - v_K_channel
    
    #if Dy is +60 mV, then at equilibrium, Kstroma/Klumen should be 10, at which point Keq=1.
    #the Keq is equal to kf/kr, so the rato of fluxes is 

    #net_Klumen=perm_K * K_Keq - perm_K/K_Keq 
    #calculate the change in lumen [K+] by multiplying the change in K+ ions
    #by the factor lumen_protons_per_turnover that relates the standard
    #complex concentration to volume:
    #the term is called "lumen_protons_per_turnover" but is the same for 
    #all species
        
    dKlumen = net_Klumen*lumen_protons_per_turnover
    
    #We assume that the stromal vaolume is large, so there should be 
    #no substantial changes in K+
    
    dKstroma=0
    #########now calculating the movement of Cl- and its impact####

    driving_force_Cl = 0.06* np.log10(Cl_stroma/Cl_lumen) + Dy
    v_VCCN1 = k_VCCN1 * Cl_flux_relative(driving_force_Cl) * (Cl_stroma + Cl_lumen)/2
    ##v_VCCN1 is rate of Cl- moving into lumen, v_CLCE is rate of Cl- moving out
    
    #here CLCE is assumed one H+ out, two Cl- comes in
    v_CLCE =  k_CLCE*(driving_force_Cl*2+pmf)*(Cl_stroma + Cl_lumen)*(Hlumen+Hstroma)/4
    #v_CLCE = k_CLCE *(Cl_lumen * Hlumen - Cl_stroma * Hstroma)
    
    net_Cl_lumen_in = v_VCCN1 + 2*v_CLCE
    dCl_lumen = net_Cl_lumen_in * lumen_protons_per_turnover
    dCl_stroma = -0.1*dCl_lumen
    
    #***************************************************************************************
    #Buffering capacity and calculation of lumen pH:
    #***************************************************************************************
    #H_leak = Per_H * ([Hlumen]-[Hstroma])
    #H_leak = 6.14e4 * (Hlumen - Hstroma)
    # Here, we convert d_protons_in into a "concentration" by dividing by the volumen
    #d_protons_leak = 6.14e4 * (Hlumen*(np.exp(39.5*Dy)) - Hstroma*np.exp(-39.5*Dy))
    #proton leak rate calculated based on P = 2 x 10^-5 cm/s ==> 6.14 e4 per PSII per s
    #39.5 = F/RT, it seems the H_leak has a relatively small impact as claimed by
    #Mordechay SchGnfeld and Hedva Schickler, FEBS letter 1983
    #The permeability of the thylakoid membrane for protons
    dHin = (net_protons_in - v_KEA - v_CLCE)*lumen_protons_per_turnover
    #It looks like earlier code did not calculate v_KEA into H+ concentrations from numbers
    #v_KEA should be numbers of ion/s across KEA. as indicated in dKlumen
    
    # Here we calculate the change in lumen pH by dividing dHin by the buffering capacity
    dpHlumen= -1*dHin / buffering_capacity 

    dHstroma = 0#(net_protons_stroma + v_KEA + v_CLCE)*lumen_protons_per_turnover/10
    #Assuming the volume of stroma is ten times as that of lumen
    dpHstroma = -1*dHstroma / buffering_capacity
    #***************************************************************************************
    #Calculation of Dy considering all ion movements and thylakoid membrane capatitance
    #***************************************************************************************
    delta_charges=charges_from_PSII+PSI_charge_separations + charges_from_b6f \
                    + d_charge_NDH - v_K_channel - d_H_ATP_or_passive - v_VCCN1- 3*v_CLCE
    #This net_Klumen does not represent the total charge caused by K+ movement
    #K+ movement only impacts charges from v_K_channel(added variable in this function)            
    #delta_charges= net_protons_in + net_Klumen # - PSII_recombination_v 
    # recall that PSII_recnotesombination is negative electrogenic 
    #note, I now inclluded this term in the calculation of PSII charge separations
    
    dDy=delta_charges*Volts_per_charge
    dpmf= 0.06* dpHlumen + dDy

    #calculate changes to deltaGatp
    #assume that deltaGatp is constant (as suggested by past resarch)...is this changes, 
    #need to consider many metabilic reactions as well.
    #Here we try to consider CBC only
    #DeltaGatp = 30.0 + 2.44* np.log(ATP_pool/ADP_pool/Pi)

    #ddeltaGatp = deltaGatp - DeltaGatp
    #d_ATP_consumed = NADPH_pool*k_CBC*(1-np.exp(-t/900))*1.5
    #if d_ATP_made - d_ATP_consumed < 0:
    #    dATP_pool = 0
    #else:
    dATP_pool= d_ATP_made - d_ATP_consumed
    dADP_pool= - dATP_pool
    #calculate changes in the concentrations of zeaxanthin (Z) and violaxanthin (V)
    #considering VDE_max_turnover_number, pKvde, VDE_Hill, kZE, and lumen pH
    
    dZ, dV = calc_v_VDE(VDE_max_turnover_number, pKvde, VDE_Hill, kZE, pHlumen, V, Z)

    #***************************************************************************************
    #The following calculated changes in NPQ based on the previous and new lumen pH
    #***************************************************************************************

    #calculate the protonation state of PsbS, considering 
    #its pKa and lumen pH
    
    new_PsbS_H = calc_PsbS_Protonation(pKPsbS, pHlumen + dpHlumen)
    new_Z=Z+dZ
    
    #calculate NPQ, based on a simple relationahip between
    #the concentration of Z and the protonation state of PsbS
    #Half contribution from Z but mostly PsbS dependent, half from PsbS alone
    new_NPQ=0.4*max_NPQ*new_PsbS_H*new_Z+0.5*max_NPQ*new_PsbS_H+0.1*max_NPQ*new_Z
    
    #feed this into odeint by calculating the change in NPQ compared to the previous
    #time point
    dNPQ=new_NPQ-NPQ #new_PsbS_H-PsbS_H

    #we re-calculate Phi2 at the start of each iteration of f, so we do not want 
    #odeint to change it
    dPhi2=0 #
    #dADP_pool= 0
    #dATP_pool = 0
    ddeltaGatp = 0
    return [dQA, dQAm, dPQ, dPQH2, dHin, dpHlumen, dDy, dpmf, ddeltaGatp, dKlumen, dKstroma, 
            d_ATP_made, d_PC_ox, d_PC_red, d_P700_ox, d_P700_red, dZ, dV, dNPQ, dsingletO2, dPhi2, dLEF, 
            dFd_ox, dFd_red,  dATP_pool, dADP_pool, dNADPH_pool,dNADP_pool, dCl_lumen, dCl_stroma,dHstroma, dpHstroma]