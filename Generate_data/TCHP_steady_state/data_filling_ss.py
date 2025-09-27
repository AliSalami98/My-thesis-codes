import csv
import numpy as np
from config import CP
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from scipy.optimize import curve_fit# Create an AbstractState object using the HEOS backend and CO2
from utils import (
	get_state,
)
data = {
    "Tfume_in [K]": [],
    "hgc_in [J/kg]": [],
    "he_in [J/kg]": [],
    "Tw_in [K]": [],

    "Tmpg_in [K]": [],
    "Tmpg_out [K]": [],

    "hc_out [J/kg]": [],
    "he_out [J/kg]": [],
    "Tc_w_in [K]": [],
    "Tc_w_out [K]": [],
    "Te_out [K]": [],
    "Tc_out [K]": [],

    "he_in [J/kg]": [],
    "pe [pa]": [],
    "p25 [pa]": [],
    "pi [pa]": [],
    "pc [pa]": [],

    "hihx1_in [J/kg]": [],
    "hihx1_out [J/kg]": [],
    "hihx2_in [J/kg]": [],
    "hihx2_out [J/kg]": [],

    "hbuff1_in [J/kg]": [],
    "hbuff1_out [J/kg]": [],
    "Tbuff1_w_in [K]": [],
    "Tbuff1_w_out [K]": [],

    "hbuff2_in [J/kg]": [],
    "hbuff2_out [J/kg]": [],
    "Tbuff2_w_in [K]": [],
    "Tbuff2_w_out [K]": [],

    "Pft [W]": [],
    "hft_in [J/kg]": [],
    "hft_g_out [J/kg]": [],
    "hft_l_out [J/kg]": [],

    "mf_dot [kg/s]": [],
    "mf1_dot [kg/s]": [],
    "mf2_dot [kg/s]": [],
    "mw_dot [kg/s]": [],
    "mw2_dot [kg/s]": [],
    "mw1_dot_1 [kg/s]": [],
    "mw1_dot_2 [kg/s]": [],
    "mCH4_dot [kg/s]": [],
    "mmpg_dot [kg/s]": [],

    "cpmpg [J/(kg.K)]": [],

    "Pc [W]": [],
    "Pe [W]": [],
    "Pihx1 [W]": [],
    "Pihx2 [W]": [],
    "Pbuff1 [W]": [],
    "Pbuff2 [W]": [],
    "Pfume [W]": [],
    "mg_dot [kg/s]": []
    
}
d_Tw_in = []
d_Theater1 = []
d_Theater2 = []
d_Theater3 = []

d_omega1 = []
d_omega2 = []
d_omega3 = []

d_pr1 = []
d_pr2 = []
d_pr3 = []

d_Hpev = []
d_Lpev = []

d_p25 = []
d_pi = []
LHV_CH4 = 50*10**6

Dw = 1 #kg/L
Dmpg = 1.04 #kg/L
DCH4 = 0.68*10**(-3) # kg/L
cpw = 4186
Apipe = np.pi/4*(12.3*10**(-3))**2

d_Tc = np.zeros((18, 2))
d_Tc_w = np.zeros((18, 2))
d_Te = np.zeros((18, 2))
d_Te_w = np.zeros((18, 2))
d_TIHX_1 = np.zeros((18, 2))
d_TIHX_2 = np.zeros((18, 2))
d_Tbuffer1 = np.zeros((18, 2))
d_Tbuffer1_w = np.zeros((18, 2))
d_Tbuffer2 = np.zeros((18, 2))
d_Tbuffer2_w = np.zeros((18, 2))

d_Cd = []
d_Delp = []
d_D_gc = []
d_D_IHX = []
d_D_evap = []
d_D_buffer1 = []
d_D_buffer2 = []
d_D_rec = []
d_AU_gc = []
d_AU_IHX = []
d_AU_evap = []
d_AU_buffer1 = []
d_AU_buffer2 = []
d_AU_rec = []
d_LMTD_gc = []
d_LMTD_IHX = []
d_LMTD_evap = []
d_LMTD_buffer1 = []
d_LMTD_buffer2 = []
d_LMTD_rec = []
d_PR1 = []
d_PR2 = []
d_PR3 = []
d_mf_dot= []
d_mw2_dot= []
d_mhp_dot = []

d_Text = []
d_Pelec = []
d_Pc = []
d_Pe = []
d_PIHX = []
d_Pcomb = []
d_Prec = []
d_Pcomp1 = []
d_Pcomp2 = []
d_Pcomp3 = []
d_Pcooler1 = []
d_Pcooler2 = []
d_Pcooler3 = []
d_Pbuffer1 = []
d_Pbuffer2 = []
d_Pbuffer3 = []
d_SH = []
d_Pin_comp = []
d_Pout_comp = []
d_Pin = []
d_Pout = []
Econs_cycle = []
Econs_comp = []
AU_rec = []
pressure_exp = np.zeros((18, 12))
enthalpy_exp = np.zeros((18, 12))

Pheat_out = []
d_COP = []
GUE = []
COP_carnot = []
SLE = []
d_Tw_return = []
d_Tw_supply = []

i = -1

d_omegab = []
d_omega1 = []
d_omega2 = []
d_omega3 = []
d_Theater1 = []
d_Theater2 = []
d_mw_dot = []
d_mmpg_dot = []
d_Tw_in = []
d_Tmpg_in = []
d_Lpev = []
d_Hpev = []
d_pc = []
d_p25 = []
d_pi = []
d_pe = []
d_Pcomb = []
d_Tw_out = []
d_Te_out = []
d_Tc_in = []
d_Tc_out = []
d_Tmpg_out = []
d_Pheat_out = []
d_mc_dot = []
d_me_dot = []
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\cycle measurements\all 2.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        i += 1
        # if i > 11:
        #     break
        T1 = float(row['T1'])+ 273.15
        T2 = float(row['T2'])+ 273.15
        T3 = float(row['T3'])+ 273.15
        T4 = float(row['T4'])+ 273.15
        T5 = float(row['T5'])+ 273.15
        T6 = float(row['T6'])+ 273.15
        T7 = float(row['T7'])+ 273.15
        T8 = float(row['T8'])+ 273.15
        T9 = float(row['T9'])+ 273.15
        T10 = float(row['T10'])+ 273.15
        T11 = float(row['T11'])+ 273.15
        T12 = float(row['T12'])+ 273.15
        T13 = float(row['T13'])+ 273.15
        T14 = float(row['T14'])+ 273.15
        T15 = float(row['T15'])+ 273.15
        T16 = float(row['T16'])+ 273.15
        T17 = float(row['T17'])+ 273.15
        T18 = float(row['T18'])+ 273.15
        T19 = float(row['T19'])+ 273.15
        T20 = float(row['T20'])+ 273.15
        T21 = float(row['T21'])+ 273.15
        T22 = float(row['T22'])+ 273.15
        T23 = float(row['T23'])+ 273.15
        T24 = float(row['T24'])+ 273.15
        T28 = max(float(row['T28'])+ 273.15, T19 + 0.2)
        Theater1 = float(row['Theater1'])+ 273.15
        Theater2 = float(row['Theater2'])+ 273.15
        Tw_return = float(row['Tw_return'])+ 273.15
        d_Tw_return.append(Tw_return)
        Tw_supply = T28 #float(row['Tw_supply'])+ 273.15
        Text = float(row['Text']) + 273.15
        d_Text.append(float(row['Text']))
        Te_out = float(row['Tevap_out'])+ 273.15
        Tc_out = float(row['Tgc_out'])+ 273.15
        pe = float(row['pe'])*10**5
        p25 = float(row['p25'])*10**5
        pi = float(row['pi'])*10**5
        pc = float(row['pc'])*10**5
        mf_dot = float(row['mf_dot'])*10**(-3) # kg/s
        d_mc_dot.append(mf_dot)
        d_mf_dot.append(mf_dot)
        mw1_dot = float(row['mw1_dot'])*Dw/60
        mw_dot = float(row['mw_dot'])*Dw/60
        mmpg_dot = float(row['mmpg_dot'])*Dmpg/60
        mCH4_dot = float(row['mCH4_dot [kg/s]'])
        data["mg_dot [kg/s]"].append(mCH4_dot * 18.125)
        mw2_dot = mw_dot - mw1_dot
        d_mw2_dot.append(mw2_dot)

        mpg_percentage = -1.136*(Text-273.15) + 43.632
        cpmpg = (100 - mpg_percentage)/100*cpw + mpg_percentage/100 * 2500
        # print(mpg_percentage)
        d_PR1.append(row['PR1'])
        d_PR2.append(row['PR2'])
        d_PR3.append(row['PR3'])
        Pelec = float(row['Pelec'])
        d_Pelec.append(Pelec)
        Pcomb = mCH4_dot*LHV_CH4

        d_Pcomb.append(Pcomb)
        d_mw_dot.append(mw_dot)
        d_mmpg_dot.append(mmpg_dot)
        d_omegab.append(float(row['omegab']))
        d_omega1.append(float(row['omega1']))
        d_omega2.append(float(row['omega2']))
        d_omega3.append(float(row['omega3']))
        d_Theater1.append(Theater1)
        d_Theater2.append(Theater2)
        d_Tw_in.append(Tw_return)
        d_Tmpg_in.append(float(row['T20'])+ 273.15)
        d_Hpev.append(float(row['Hpev']))
        d_Lpev.append(float(row['Lpev']))
        d_pc.append(pc)
        d_p25.append(p25)
        d_pi.append(pi)
        d_pe.append(pe)
        d_Tc_in.append(T11)
        d_Tc_out.append(Tc_out)
        d_Te_out.append(Te_out)
        state = get_state(CP.PQ_INPUTS, pe, 1)
        d_SH.append(Te_out - state.T())
        d_Tmpg_out.append(float(row['T21'])+ 273.15)
        d_Tw_out.append(Tw_supply)

        # -------------------------------
        # Recovery heat exchanger
        # -------------------------------
        Prec = max(mw_dot*cpw*(T28 - T19), 100)
        d_Prec.append(Prec)
        data['Tfume_in [K]'].append(T13)
        AU_rec.append(Prec/(T13 - T19))
        # LMTD_rec.append()
        # ------------------------------
        # Gas cooler
        # ------------------------------
        Pc_w = mw2_dot*cpw*(T18-T9)
        d_Pc.append(Pc_w*10**(-3))
        state = get_state(CP.PT_INPUTS, pc, T22)
        h22 = state.hmass()
        state = get_state(CP.PT_INPUTS, pc, T11)
        h11 = state.hmass()
        D11 = state.rhomass()
        hc_out = h11 - Pc_w/mf_dot
        state = get_state(CP.HmassP_INPUTS, hc_out, pc)
        quality_vapor = state.Q()
        # if 0 < quality_vapor < 1:
        #     continue
        # Tc_out = state.T()

        DTc_1 = np.abs(T11 - T18)
        DTc_2 = np.abs(T22 - T9)
        LMTD_gc = (DTc_1 - DTc_2)/np.log(DTc_1/DTc_2)
        d_LMTD_gc.append(LMTD_gc)
        d_AU_gc.append(Pc_w/LMTD_gc)
        d_D_gc.append(D11)

        d_Tc[i][:] = [T11-273.15, T22-273.15]
        d_Tc_w[i][:] = [T18-273.15, T9-273.15]
        # ------------------------------
        # Tank
        # ------------------------------
        state = get_state(CP.PQ_INPUTS, p25, 0)
        hliq_tank = state.hmass() #CP.PropsSI('H', 'P', p25, 'Q', 0, 'CO2') # enthalpy of liquid leaving tank
        state = get_state(CP.PT_INPUTS, p25, T7)
        h7 = state.hmass() #CP.PropsSI('H', 'T', T8, 'Q', 1, 'CO2')  #  enthalpy of vapor leaving tank
        # ------------------------------
        # Lp valve
        # ------------------------------
        h24 = hliq_tank
        state = get_state(CP.HmassP_INPUTS, h24, p25)
        D24 = state.rhomass()

        # m1_dot = Apipe*Cd1*np.sqrt(2*D24*(p25 - pe))
        # ------------------------------
        # Evap
        # ------------------------------
        state = get_state(CP.PT_INPUTS, pe, Te_out)
        he_out = state.hmass()
        Pe = mmpg_dot*cpmpg*(T20 - T21)
        d_Pe.append(Pe)
        mf1_dot = Pe/(he_out - h24)
        d_me_dot.append(mf1_dot)
        DTe_1 = np.abs(T20 - Te_out)
        DTe_2 = np.abs(T21 - T24)
        LMTD_evap = (DTe_1 - DTe_2)/np.log(DTe_1/DTe_2)
        d_LMTD_evap.append(LMTD_evap)
        d_AU_evap.append(Pe/LMTD_evap)

        d_Te[i][:] = [T24-273.15, Te_out-273.15]
        d_Te_w[i][:] = [T21-273.15, T20-273.15]
        # ------------------------------
        # IHX
        # ------------------------------
        state = get_state(CP.PT_INPUTS, pe, T1)
        h1 = state.hmass()
        h22 = hc_out #CP.PropsSI('H', 'P', pc, 'T', T22, 'CO2')
        PIHX = mf1_dot*(h1 - he_out) # = mf_dot*(h22 - h23)
        d_PIHX.append(PIHX)
        h23 = h22 - PIHX/mf_dot

        DTIHX_1= np.abs(T22 - T1)
        DTIHX_2 = np.abs(T23 - Te_out)
        LMTD_IHX = (DTIHX_1 - DTIHX_2)/np.log(DTIHX_1/DTIHX_2)
        d_LMTD_IHX.append(LMTD_IHX)
        d_AU_IHX.append(PIHX/LMTD_IHX)

        d_TIHX_1[i][:] = [T22-273.15, T23-273.15]
        d_TIHX_2[i][:] = [T1-273.15, Te_out-273.15]
        # ------------------------------
        # Hp valve
        # ------------------------------
        h8 = h23
        # state = get_state(CP.HmassP_INPUTS, h23, pc)
        # D23 = state.rhomass()

        # ------------------------------
        # comp 2
        # ------------------------------
        state = get_state(CP.PT_INPUTS, p25, T3)
        h3 = state.hmass()
        state = get_state(CP.PT_INPUTS, pi, T4)
        h4 = state.hmass()
        Pcomp2 = mf_dot*(h4 - h3)
        d_Pcomp2.append(Pcomp2)
        Pcooler2 = mw1_dot*cpw*(T16 - T15)
        d_Pcooler2.append(Pcooler2)
        # ------------------------------
        # buffer 2
        # ------------------------------
        state = get_state(CP.PT_INPUTS, pi, T5)
        h5 = state.hmass()
        # Pbuffer2_w = mw1_dot_2*cpw*(T12 - T10)
        Pbuffer2 = mf_dot *(h4 - h5)
        d_Pbuffer2.append(Pbuffer2)
        mw1_dot_2 = max(Pbuffer2/(cpw*(T12 - T10)), 0)
        mw1_dot_1 = mw1_dot - mw1_dot_2
        DTbuffer2_1 = np.abs(T4 - T10)  #parallel flow
        DTbuffer2_2 = np.abs(T5 - T12)
        LMTD_buffer2 = (DTbuffer2_1 - DTbuffer2_2)/np.log(DTbuffer2_1/DTbuffer2_2)
        d_LMTD_buffer2.append(LMTD_buffer2)
        d_AU_buffer2.append(Pbuffer2/LMTD_buffer2)

        d_Tbuffer2[i][:] = [T4-273.15, T5-273.15]
        d_Tbuffer2_w[i][:] = [T10-273.15, T12-273.15]
        # ------------------------------
        # buffer 1
        # ------------------------------        
        # ------------------------------
        # Connection (buffer 1 X tank X comp2)
        # ------------------------------
        state = get_state(CP.PT_INPUTS, p25, T2)
        h2 = state.hmass()
        mf2_dot = mf_dot - mf1_dot
        hbuffer1_out = (mf_dot*h3 - mf2_dot*h7)/mf1_dot
        # Pbuffer1_w = mw1_dot_1*cpw*(Tw_buffer1_out - T9)
        state = get_state(CP.HmassP_INPUTS, hbuffer1_out, p25)
        Tbuffer1_out = state.T()
        Pbuffer1 = mf1_dot *(h2 - hbuffer1_out)
        d_Pbuffer1.append(Pbuffer1)
        Tw_buffer1_out = Pbuffer1/(mw1_dot_1*cpw) + T9

        DTbuffer1_1 = np.abs(T2 - T9)  #parallel flow
        DTbuffer1_2 = np.abs(Tbuffer1_out - Tw_buffer1_out)
        LMTD_buffer1 = (DTbuffer1_1 - DTbuffer1_2)/np.log(DTbuffer1_1/DTbuffer1_2)
        d_LMTD_buffer1.append(LMTD_buffer1)
        d_AU_buffer1.append(Pbuffer1/LMTD_buffer1)

        d_Tbuffer1[i][:] = [T2-273.15, Tbuffer1_out-273.15]
        d_Tbuffer1_w[i][:] = [T9-273.15, Tw_buffer1_out-273.15]
        # ------------------------------
        # comp 1
        # ------------------------------
        Pcomp1 = mf1_dot*(h2 - h1)
        d_Pcomp1.append(Pcomp1)
        Pcooler1 = mw1_dot*cpw*(T15 - T14)
        d_Pcooler1.append(Pcooler1)

        # ------------------------------
        # comp 3
        # ------------------------------
        state = get_state(CP.PT_INPUTS, pc, T6)
        h6 = state.hmass()
        Pcomp3 = mf_dot*(h6 - h5)
        d_Pcomp3.append(Pcomp3)
        Pcooler3 = mw1_dot*cpw*(T17 - T16)
        d_Pcooler3.append(Pcooler3)
        # ------------------------------
        # buffer 3
        # ------------------------------
        Pbuffer3 = mf_dot *(h6 - h11)
        d_Pbuffer3.append(Pbuffer3)
        
        # pressure_exp[i][:] = [p25, p25, p25, pi, pi, pc, pc, pc, pc, p25, p25, pe, pe, pe, p25, p25]        
        # pressure_exp[i][:] = [x*10**(-5) for x in pressure_exp[i][:]]

        # enthalpy_exp[i][:] = [h8, h7, h3, h4, h5, h6, h11, h22, h23, h8, h24, h24, he_out, h1, h2, h3]
        # enthalpy_exp[i][:] = [x*10**(-3) for x in enthalpy_exp[i][:]]

        pressure_exp[i][:] = [p25, p25, pc, pc, pc, p25, p25, pe, pe, pe, p25, p25]        
        pressure_exp[i][:] = [x*10**(-5) for x in pressure_exp[i][:]]

        enthalpy_exp[i][:] = [h8, h3, h11, h22, h23, h8, h24, h24, he_out, h1, h2, h3]
        enthalpy_exp[i][:] = [x*10**(-3) for x in enthalpy_exp[i][:]]

        Theater = float(row['Theater1']) + 273.15
        Tw_return = float(row['Tw_return']) + 273.15
        Tw_supply = float(row['Tw_supply']) + 273.15

        Text = float(row['Text']) + 273.15

        data['hgc_in [J/kg]'].append(h11)
        data['hc_out [J/kg]'].append(hc_out)
        data['he_in [J/kg]'].append(h24)
        data['he_out [J/kg]'].append(he_out)
        data['hbuff1_in [J/kg]'].append(h2)
        data['hbuff1_out [J/kg]'].append(hbuffer1_out)
        data['hbuff2_in [J/kg]'].append(h4)
        data['hbuff2_out [J/kg]'].append(h5)
        data['hihx1_in [J/kg]'].append(hc_out)
        data['hihx1_out [J/kg]'].append(h23)
        data['hihx2_in [J/kg]'].append(he_out)
        data['hihx2_out [J/kg]'].append(h1)
        data['hft_in [J/kg]'].append(h23)
        data['hft_l_out [J/kg]'].append(h24)
        data['hft_g_out [J/kg]'].append(h7)

        data['Tmpg_in [K]'].append(float(row['T20'])+ 273.15)
        data['Tmpg_out [K]'].append(float(row['T21'])+ 273.15)
        data['Tw_in [K]'].append(float(row['T9'])+ 273.15)
        data['Tbuff1_w_in [K]'].append(float(row['T10'])+ 273.15)
        data['Tbuff1_w_out [K]'].append(float(row['T12'])+ 273.15)
        data['Tc_w_out [K]'].append(float(row['T18'])+ 273.15)
        data['Tc_out [K]'].append(Tc_out)
        data['Te_out [K]'].append(Te_out)


        data['pe [pa]'].append(float(row['pe'])*10**5)
        data['p25 [pa]'].append(float(row['p25'])*10**5)
        data['pi [pa]'].append(float(row['pi'])*10**5)
        data['pc [pa]'].append(float(row['pc'])*10**5)

        data['mf_dot [kg/s]'].append(mf_dot) # kg/s
        data['mf1_dot [kg/s]'].append(mf1_dot)

        data['mw2_dot [kg/s]'].append(mw2_dot)
        data['mw1_dot_1 [kg/s]'].append(mw1_dot_1)
        data['mw1_dot_2 [kg/s]'].append(mw1_dot_2)
        data['mw_dot [kg/s]'].append(float(row['mw_dot'])*Dw/60)
        data['mmpg_dot [kg/s]'].append(float(row['mmpg_dot'])*Dmpg/60)
        data['mCH4_dot [kg/s]'].append(float(row['mCH4_dot [kg/s]']))

        data['cpmpg [J/(kg.K)]'].append(cpmpg)

        data['Pc [W]'].append(Pc_w)
        data['Pe [W]'].append(Pe)
        data['Pihx1 [W]'].append(PIHX)
        data['Pihx2 [W]'].append(PIHX)
        data['Pbuff1 [W]'].append(Pbuffer1)
        data['Pbuff2 [W]'].append(Pbuffer2)
        data['Pfume [W]'].append(Prec)

        d_Theater3.append((T13))
        # d_COP.append(mw_dot*cpw*(Tw_supply - Tw_return)/Pcomb)
        d_COP.append((Pc_w + Pcooler1 + Pcooler2 + Pcooler3 + Pbuffer1 + Pbuffer2 + Prec)/Pcomb)
        d_Pheat_out.append(Pc_w + Pcooler1 + Pcooler2 + Pcooler3 + Pbuffer1 + Pbuffer2 + Prec)
        # print(Pbuffer2)



# AUrec = []
# for i in range(len(data['Tfume_in [K]'])):

#     deltd_T_lm = (data['Tfume_in [K]'][i] - data['Tw_in [K]'][i]) #(deltd_T1 - deltd_T2) / np.log(deltd_T1 / deltd_T2

#     # Heat transfer rate (Q)
#     Q = data['Pfume [W]'][i]

#     # Calculate AUrec
#     if deltd_T_lm > 0:
#         AU_value = Q / deltd_T_lm
#         AUrec.append(AU_value)
#     else:
#         AUrec.append(0)

# from mpl_toolkits.mplot3d import Axes3D

# # Convert mass flow rates and AUrec to NumPy arrays
# m_gas = np.array(data['mg_dot [kg/s]'])  # Gas mass flow rate
# m_water = np.array(data['mw_dot [kg/s]'])  # Water mass flow rate
# AUrec = np.array(AUrec)                   # AUrec values

# # Define a bivariate linear model
# def bivariate_model(vars, a, b, c):
#     m_gas, m_water = vars
#     return a * m_gas + b * m_water + c

# # Prepare input variables for curve fitting
# vars = (m_gas, m_water)

# # Fit the model
# params, _ = curve_fit(bivariate_model, vars, AUrec)

# # Generate fitted values
# AUrec_fitted = bivariate_model(vars, *params)

# # Print model parameters
# print("Fitted Parameters:")
# print(f"a (gas flow rate coefficient) = {params[0]}")
# print(f"b (water flow rate coefficient) = {params[1]}")
# print(f"c (intercept) = {params[2]}")

# # 3D Plot to visualize the fit
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(m_gas, m_water, AUrec, label='Original Data', color='blue')
# ax.scatter(m_gas, m_water, AUrec_fitted, label='Fitted Data', color='red', alpha=0.6)
# ax.set_xlabel('Gas Mass Flow Rate (kg/s)')
# ax.set_ylabel('Water Mass Flow Rate (kg/s)')
# ax.set_zlabel('AUrec (W/K)')
# ax.set_title('AUrec as Function of Gas and Water Mass Flow Rates')
# ax.legend()
# plt.show()

# # 2D Contour Plot
# plt.figure(figsize=(10, 6))
# plt.tricontourf(m_gas, m_water, AUrec_fitted, levels=20, cmap='viridis')
# plt.colorbar(label='AUrec (W/K)')
# plt.scatter(m_gas, m_water, c='red', label='Data Points')
# plt.xlabel('Gas Mass Flow Rate (kg/s)')
# plt.ylabel('Water Mass Flow Rate (kg/s)')
# plt.title('Contour Plot of Fitted AUrec')
# plt.legend()
# plt.show()

# # Fit and plot AUrec as before
# x = np.array(data['mw_dot [kg/s]'])  # Example: Use water flow rate as the independent variable
# AUrec = np.array(AUrec)

# # Define linear fit model
# def linear_model(x, a, b):
#     return a * x + b

# # Fit the linear model
# params, _ = curve_fit(linear_model, x, AUrec)

# # Generate fitted AUrec
# AUrec_fitted = linear_model(x, *params)

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.scatter(x, AUrec, label="Original AUrec Data", color="blue")
# plt.plot(x, AUrec_fitted, label="Linear Fit", color="red")
# plt.xlabel("Mass Flow Rate of Water (kg/s)")
# plt.ylabel("AUrec (W/K)")
# plt.title("AUrec vs. Water Mass Flow Rate")
# plt.legend()
# plt.grid()
# plt.show()

# # Print linear fit parameters
# print("Linear Fit Parameters: a =", params[0], ", b =", params[1])
