import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import csv
import matplotlib.pyplot as plt

# Create an AbstractState object using the HEOS backend and CO2
state = AbstractState("HEOS", "CO2")

LHV_CH4 = 50*10**6

Dw = 1 #kg/L
Dmpg = 1.04 #kg/L
DCH4 = 0.68*10**(-3) # kg/L
cpw = 4186
cpmpg = 0.7*4186 + 0.3*2500
Apipe = np.pi/4*(12.3*10**(-3))**2


a_T11 = []
a_T22 = []
a_T9 = []
a_T17 = []

a_T18 = []
a_T24 = []
a_T20 = []
a_T19 = []
a_T28 = []

a_T21 = []
a_T23 = []
a_T1 = []
a_T2 = []
a_T3 = []
a_T4 = []
a_T5 = []
a_T14 = []
a_T15 = []
a_T6 = []
a_T10 = []
a_T12 = []
a_T13 = []
a_Tbuffer1_out = []
a_Tw_buffer1_out = []
a_Theater = []
a_pevap = []
a_p25 = []
a_pi = []
a_pgc = []
a_h1 = []
a_h2 = []
a_h3 = []
a_h4 = []
a_h5 = []
a_h6 = []
a_h7 = []
a_h11 = []
a_h22 = []
a_hgc_out = []
a_h23 = []
a_h8 = []
a_h24 = []
a_hevap_out = []
a_hbuffer1_out = []
a_mw_dot = []
a_mw1_dot = []
a_mw_dot_buffer1 = []
a_mw_dot_buffer2 = []
a_mmpg_dot = []
a_Tevap_out = []

a_Cd = []
a_Delp = []
a_D_gc = []
a_D_IHX = []
a_D_evap = []
a_D_buffer1 = []
a_D_buffer2 = []
a_D_fhx = []
a_AU_gc = []
a_AU_IHX = []
a_AU_evap = []
a_AU_buffer1 = []
a_AU_buffer2 = []
a_AU_fhx = []
a_LMTD_gc = []
a_LMTD_IHX = []
a_LMTD_evap = []
a_LMTD_buffer1 = []
a_LMTD_buffer2 = []
a_LMTD_fhx = []
a_PR1 = []
a_PR2 = []
a_PR3 = []
a_mf1_dot = []
a_mf_dot= []
a_mw2_dot= []
a_mbp_dot = []
a_mhp_dot = []
a_Hpev = []
a_Lpev = []
a_DpHp = []
a_DpLp = []
a_mCH4_dot = []
a_Text = []
a_Pelec = []
a_Pgc = []
a_Pevap = []
a_PIHX = []
a_Pcomb = []
a_Pfhx = []
a_Pcomp1 = []
a_Pcomp2 = []
a_Pcomp3 = []
a_Pcooler1 = []
a_Pcooler2 = []
a_Pcooler3 = []
a_Pbuffer1 = []
a_Pbuffer2 = []
a_Pbuffer3 = []
a_PHXs_t = []
a_Pin_comp = []
a_Pout_comp = []
a_Pin = []
a_Pout = []
Econs_cycle = []
Econs_comp = []

Pheat_out = []
COP = []
GUE = []
COP_carnot = []
SLE = []
exergy = []
Res_HP = []
Res_water = []

Res_HP_min = []
Res_HP_max = []
Res_TCs = []
Res_ratio = []
f_HPV = []

i = 0
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\cycle measurements\all 2.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        if i >= 0:
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
            Theater = float(row['Theater1'])+ 273.15
            Tw_return = T9
            Tw_supply = T28 #float(row['Tw_supply'])+ 273.15
            a_T19.append(T19)
            a_T28.append(T28)
            Text = float(row['Text']) + 273.15
            Tevap_out = float(row['Tevap_out'])+ 273.15
            Tgc_out = float(row['Tgc_out'])+ 273.15
            pevap = float(row['pe'])*10**5
            p25 = float(row['p25'])*10**5
            pi = float(row['pi'])*10**5
            pgc = float(row['pc'])*10**5
            mf_dot = float(row['mf_dot'])*10**(-3) # kg/s
            mw1_dot = float(row['mw1_dot'])*Dw/60
            mw_dot = float(row['mw_dot'])*Dw/60
            mCH4_dot = float(row['mCH4_dot [kg/s]'])
            mw2_dot = mw_dot - mw1_dot
            mmpg_dot = float(row['mmpg_dot'])*Dmpg/60
            mpg_percentage = -1.136*(Text-273.15) + 43.632
            cpmpg = (100 - mpg_percentage)/100*cpw + mpg_percentage/100 * 2500
            Pelec = float(row['Pelec'])
            Hpev = float(row['Hpev'])
            Lpev = float(row['Lpev'])
            a_mCH4_dot.append(mCH4_dot)
            # -------------------------------
            # Combustion
            # -------------------------------
            Pcomb = float(row['Pcomb'])
            # -------------------------------
            # fhxovery heat exchanger
            # -------------------------------
            Pfhx = mw_dot*cpw*(T28 - T19)
            # ------------------------------
            # Gas cooler
            # ------------------------------
            h22 = CP.PropsSI('H', 'P', pgc, 'T', T22, 'CO2')
            Pgc_w = mw2_dot*cpw*(T18-T9)
            h11 = CP.PropsSI('H', 'P', pgc, 'T', T11, 'CO2')
            D11 = CP.PropsSI('D', 'P', pgc, 'T', T11, 'CO2')
            hgc_out = h11 - Pgc_w/mf_dot
            # print(hgc_out - h22)
            a_hgc_out.append(hgc_out)
            # mf_dot = Pgc_w/(h11 - h22)
            quality_vapor = CP.PropsSI('Q', 'P', pgc, 'H', hgc_out, 'CO2')
            Tgc_out = CP.PropsSI('T', 'P', pgc, 'H', hgc_out, 'CO2')

            # ------------------------------
            # Tank
            # ------------------------------
            hft_l = CP.PropsSI('H', 'P', p25, 'Q', 0, 'CO2') #CP.PropsSI('H', 'P', p25, 'Q', 0, 'CO2') # enthalpy of liquid leaving tank
            h7 = CP.PropsSI('H', 'P', p25, 'T', T7, 'CO2') #CP.PropsSI('H', 'T', T8, 'Q', 1, 'CO2')  #  enthalpy of vapor leaving tank
            # ------------------------------
            # Lp valve
            # ------------------------------
            h24 = hft_l
            D24 = CP.PropsSI('Dmass', 'P', p25, 'H', h24, 'CO2')
            # m1_dot = Apipe*Cd1*np.sqrt(2*D24*(p25 - pevap))
            # ------------------------------
            # Evap
            # ------------------------------
            hevap_out = CP.PropsSI('H', 'P', pevap, 'T', Tevap_out, 'CO2')
            Pevap = mmpg_dot*cpmpg*(T20 - T21)
            mf1_dot = Pevap/(hevap_out - h24)

            # print(mf1_dot)
            # ------------------------------
            # IHX
            # ------------------------------
            h1 = CP.PropsSI('H', 'P', pevap, 'T', T1, 'CO2')
            h22 = CP.PropsSI('H', 'P', pgc, 'T', T22, 'CO2')

            PIHX = mf1_dot*(h1 - hevap_out) # = mf_dot*(h22 - h23)
            h23 = h22 - PIHX/mf_dot
            # ------------------------------
            # Hp valve
            # ------------------------------
            h8 = h23
            D23 = CP.PropsSI('Dmass', 'P', pgc, 'H', h23, 'CO2')

            # ------------------------------
            # comp 2
            # ------------------------------
            h3 = CP.PropsSI('H', 'P', p25, 'T', T3, 'CO2')
            h4 = CP.PropsSI('H', 'P', pi, 'T', T4, 'CO2')
            Pcomp2 = mf_dot*(h4 - h3)
            Pcooler2 = mw1_dot*cpw*(T16 - T15)
            # ------------------------------
            # buffer 2
            # ------------------------------
            h5 = CP.PropsSI('H', 'P', pi, 'T', T5, 'CO2')
            # Pbuffer2_w = mw1_dot_2*cpw*(T12 - T10)
            Pbuffer2 = mf_dot *(h4 - h5)

            mw1_dot_2 = max(Pbuffer2/(cpw*(T12 - T10)), 0)
            mw1_dot_1 = mw1_dot - mw1_dot_2
            a_mw_dot_buffer1.append(mw1_dot_1)
            a_mw_dot_buffer2.append(mw1_dot_2)
            # ------------------------------
            # buffer 1
            # ------------------------------ 
            h2 = CP.PropsSI('H', 'P', p25, 'T', T2, 'CO2')
            mf2_dot = mf_dot - mf1_dot
            hbuffer1_out = (mf_dot*h3 - mf2_dot*h7)/mf1_dot
            # Tw_buffer1_out = (mw1_dot*T14 - mw1_dot_2*T12)/mw1_dot_1 
            # Pbuffer1_w = mw1_dot_1*cpw*(Tw_buffer1_out - T9)
            # hbuffer1_out = h2 - Pbuffer1_w/mf1_dot
            Tbuffer1_out = CP.PropsSI('T', 'H', hbuffer1_out, 'P', p25, 'CO2')
            Pbuffer1 = mf1_dot *(h2 - hbuffer1_out)
            # h7_ = (mf_dot*h3 - mf1_dot*hbuffer1_out)/mf2_dot

            # print(mw1_dot_1/mw1_dot)
            Tw_buffer1_out = T9 + Pbuffer1/(cpw * mw1_dot_1)

            # ------------------------------
            # comp 1
            # ------------------------------
            Pcomp1 = mf1_dot*(h2 - h1)
            Pcooler1 = mw1_dot*cpw*(T15 - T14)
            # ------------------------------
            # comp 3
            # ------------------------------
            h6 = CP.PropsSI('H', 'P', pgc, 'T', T6, 'CO2')
            Pcomp3 = mf_dot*(h6 - h5)
            Pcooler3 = mw1_dot*cpw*(T17 - T16)
            # ------------------------------
            # buffer 3
            # ------------------------------
            Pbuffer3 = mf_dot *(h6 - h11)
            DTgc_1 = np.abs(T11 - T18)
            DTgc_2 = np.abs(T22 - T9)
            LMTD_gc = (T11 + T22 - (T9 + T18))/2 #(DTgc_1 - DTgc_2)/np.log(DTgc_1/DTgc_2)

            DTevap_1 = np.abs(T20 - Tevap_out)
            DTevap_2 = np.abs(T21 - T24)
            LMTD_evap = (DTevap_1 - DTevap_2)/np.log(DTevap_1/DTevap_2) #(T20 + T21 - (T24 + Tevap_out))/2 #

            DTIHX_1= np.abs(T22 - T1)
            DTIHX_2 = np.abs(T23 - Tevap_out)
            LMTD_IHX = (T22 + T23 - (T1 + Tevap_out))/2 #(DTIHX_1 - DTIHX_2)/np.log(DTIHX_1/DTIHX_2)
        
            DTbuffer2_1 = np.abs(T4 - T10)  #parallel flow
            DTbuffer2_2 = np.abs(T5 - T12)
            LMTD_buffer2 = (T4 + T5 - (T10 + T12))/2 #(DTbuffer2_1 - DTbuffer2_2)/np.log(DTbuffer2_1/DTbuffer2_2)

            DTbuffer1_1 = np.abs(T2 - T9)  #parallel flow
            DTbuffer1_2 = np.abs(Tbuffer1_out - Tw_buffer1_out)
            LMTD_buffer1 = (Tbuffer1_out + T2 - (T9 + Tw_buffer1_out))/2 #(DTbuffer1_1 - DTbuffer1_2)/np.log(DTbuffer1_1/DTbuffer1_2)
            
            a_T11.append(T11)
            a_T22.append(T22)
            a_T3.append(T3)
            a_T6.append(T6)
            a_T9.append(T9)
            a_T18.append(T18)
            a_T24.append(T24)
            a_Tevap_out.append(Tevap_out)
            a_T20.append(T20)
            a_T21.append(T21)
            a_T23.append(T23)
            a_T1.append(T1)
            a_T2.append(T2)
            a_T4.append(T4)
            a_T5.append(T5)
            a_T14.append(T14)
            a_T15.append(T15)
            a_T17.append(T17)
            a_T10.append(T10)
            a_T12.append(T12)
            a_Tbuffer1_out.append(Tbuffer1_out)
            a_Tw_buffer1_out.append(Tw_buffer1_out)
            a_Theater.append(Theater)
            a_T13.append(T13)
            a_pevap.append(pevap)
            a_p25.append(p25)
            a_pi.append(pi)
            a_pgc.append(pgc)
            a_h1.append(h1)
            a_h2.append(h2)
            a_h3.append(h3)
            a_h4.append(h4)
            a_h5.append(h5)
            a_h6.append(h6)
            a_h7.append(h7)
            a_h11.append(h11)
            a_h22.append(h22)
            a_h23.append(h23)
            a_h8.append(h8)
            a_h24.append(h24)
            a_hevap_out.append(hevap_out)
            a_hbuffer1_out.append(hbuffer1_out)

            a_Text.append(Text)
            a_mw2_dot.append(mw2_dot)
            a_PR1.append(row['PR1'])
            a_PR2.append(row['PR2'])
            a_PR3.append(row['PR3'])
            a_LMTD_gc.append(LMTD_gc)
            a_AU_gc.append(Pgc_w/LMTD_gc)
            a_D_gc.append(D11)
            a_LMTD_evap.append(LMTD_evap)
            a_AU_evap.append(Pevap/LMTD_evap)
            a_LMTD_IHX.append(LMTD_IHX)
            a_AU_IHX.append(PIHX/LMTD_IHX)
            
            # if Lpev > 50:
            a_mf_dot.append(mf_dot)
            a_mf1_dot.append(mf1_dot)
            a_mhp_dot.append(Hpev*np.sqrt(2*D23*(pgc - p25)))
            a_mbp_dot.append(Lpev*np.sqrt(2*D24*(p25 - pevap)))
            f_HPV.append(mf_dot/np.sqrt(2*D23*(pgc - p25)))
            a_Hpev.append(Hpev)
            a_Lpev.append(Lpev)
            a_DpHp.append(pgc - p25)
            a_DpLp.append(p25 - pevap)

            a_Delp.append(pgc-p25)
            # a_Cd.append(mf_dot/a_mhp_dot[-1])
            a_LMTD_buffer2.append(LMTD_buffer2)
            a_AU_buffer2.append(Pbuffer2/LMTD_buffer2)
            a_LMTD_buffer1.append(LMTD_buffer1)
            a_AU_buffer1.append(Pbuffer1/LMTD_buffer1)


            # -------------------------------------
            # Energy and Exergy evaluation
            # ------------------------------------
            a_Pcomb.append(Pcomb)
            a_Pfhx.append(Pfhx)
            a_Pgc.append(Pgc_w)
            a_Pevap.append(Pevap)
            a_PIHX.append(PIHX)
            a_Pcomp2.append(Pcomp2)
            a_Pcooler2.append(Pcooler2)
            a_Pbuffer2.append(Pbuffer2)
            a_Pbuffer1.append(Pbuffer1)
            a_Pcomp1.append(Pcomp1)
            a_Pcooler1.append(Pcooler1)
            a_Pcomp3.append(-Pcomp3)
            a_Pcooler3.append(Pcooler3)
            a_Pbuffer3.append(Pbuffer3)
            a_PHXs_t.append(Pcomb - Pgc_w) #Pcooler1 + Pcooler2 + Pcooler3 + Pbuffer1 + Pbuffer2 + Pfhx)
            a_Pelec.append(Pelec)
            Pin_comp = Pcomb
            a_Pin_comp.append(Pin_comp)
            Pout_comp = Pcomp1 + Pcomp2 + Pcomp3
            a_Pout_comp.append(Pout_comp)
            Econs_comp.append(Pin_comp - Pout_comp)

            Pin = Pevap + Pcomp1 + Pcomp2 + Pcomp3 + Pbuffer1 + Pbuffer2 + Pbuffer3
            a_Pin.append(Pin)
            Pout = Pgc_w + PIHX
            a_Pout.append(Pout)
            Econs_cycle.append(Pin - Pout)

            Pheat_out.append(mw_dot*cpw*(Tw_supply - Tw_return))
            COP.append(mw_dot*cpw*(Tw_supply - Tw_return)/Pelec)
            GUE.append(mw_dot*cpw*(Tw_supply - Tw_return)/Pcomb)
            COP_carnot.append((1-Tw_supply/Theater)*Tw_supply/(Tw_supply - Text))
            SLE.append(mw_dot*cpw*(Tw_supply - Tw_return)/Pcomb * (1 - Text/Tw_supply)/(1 - Tw_supply/Theater))
            exergy.append(Pcomb*(1 - Text/Theater))
            Res_HP.append((Pcomp1 + Pcomp2 + Pcomp3 + mf2_dot * (h3 - h8) + mf1_dot* (h1 - h24) - Pbuffer1 - Pbuffer2 - mf_dot * (h6 - h23) - mf1_dot * (h8 - hft_l))/(Pcomp1 + Pcomp2 + Pcomp3 + mf2_dot * (h3 - h8) + mf1_dot* (h1 - h24)))
            # Res_HP_min.append((0.97 * Pcomb + 0.9766 * Pevap + 0.975 * Pelec) -  1.0234 * mw_dot*cpw*(Tw_supply - Tw_return))
            # Res_HP_max.append((1.03 * Pcomb + 1.0234 * Pevap + 1.025 * Pelec) -  0.9766 * mw_dot*cpw*(Tw_supply - Tw_return))
            Heat_loss = 0 #0.1 * Pcomb
            Res_TCs.append((Pcomb + Pelec - (Pcomp1 + Pcomp2 + Pcomp3) - (Pcooler1 + Pcooler2 + Pcooler3) - Pfhx - Heat_loss)/Pcomb)
            Res_water.append(((Heat_loss + Pgc_w + mw1_dot * cpw * (T17 - T9) + Pfhx) -  mw_dot * cpw * (Tw_supply - Tw_return))/(Heat_loss + Pgc_w + mw1_dot * cpw * (T17 - T9) + Pfhx))
            Res_ratio.append(Res_TCs[-1]/Pcomb)
            a_mw_dot.append(mw_dot)
            a_mmpg_dot.append(mmpg_dot)
            a_mw1_dot.append(mw1_dot)
            # Res_cycle.append()



        i = i+1

# print(a_Pbuffer2)
from scipy.optimize import curve_fit

def heat_transfer_model(T, UA):
    return UA * T

Hpev, pcov = curve_fit(heat_transfer_model, a_mhp_dot, a_mf_dot)
Hpev_coef = Hpev[0]

bpev, pcov = curve_fit(heat_transfer_model, a_mbp_dot, a_mf1_dot)
bpev_coef = bpev[0]
# print(Hpev_coef)
# print(bpev_coef)

# fig1 = plt.figure(1)
# plt.scatter(a_mf_dot, [Hpev_coef*x for x in a_mhp_dot])
# plt.plot(a_mf_dot,a_mf_dot)

# fig2 = plt.figure(2)
# plt.scatter(a_mf1_dot, [bpev_coef*x for x in a_mbp_dot])
# plt.plot(a_mf1_dot,a_mf1_dot)
# plt.show()


# popt_gc, pcov = curve_fit(heat_transfer_model, a_LMTD_gc, a_Pgc)
# optimized_UA_gc = popt_gc[0]

# popt_evap, pcov = curve_fit(heat_transfer_model, a_LMTD_evap, a_Pevap)
# optimized_UA_evap = popt_evap[0]

# popt_IHX, pcov = curve_fit(heat_transfer_model, a_LMTD_IHX, a_PIHX)
# optimized_UA_IHX = popt_IHX[0]

# popt_buffer1, pcov = curve_fit(heat_transfer_model, a_LMTD_buffer1, a_Pbuffer1)
# optimized_UA_buffer1 = popt_buffer1[0]

# popt_buffer2, pcov = curve_fit(heat_transfer_model, a_LMTD_buffer2, a_Pbuffer2)
# optimized_UA_buffer2 = popt_buffer2[0]

# print(f"Optimized UA_gc = {optimized_UA_gc:.2f} W/K")
# print(f"Optimized UA_evap = {optimized_UA_evap:.2f} W/K")
# print(f"Optimized UA_IHX = {optimized_UA_IHX:.2f} W/K")
# print(f"Optimized UA_buffer1 = {optimized_UA_buffer1:.2f} W/K")
# print(f"Optimized UA_buffer2 = {optimized_UA_buffer2:.2f} W/K")
# # ''''''''''''''''''''''''''''''''''''''''''''''
# # ML optimization
# #''''''''''''''''''''''''''''''''''''''''''''''''

# x1 = np.array(a_mbp_dot).reshape(-1, 1)
# y1 = np.array(a_mf1_dot)

# from sklearn.model_selection import train_test_split
# X1_train, X1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = 0.2, random_state = 0)

# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression

# poly1_reg = PolynomialFeatures(degree = 1)
# X1_poly = poly1_reg.fit_transform(X1_train)
# model1_PR = LinearRegression()
# model1_PR.fit(X1_poly, y1_train)

# y1_PR_pred = model1_PR.predict(poly1_reg.transform(X1_test))

# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
# mape1_PR = mean_absolute_percentage_error(y1_test, y1_PR_pred) *100

# # print(mape1_PR)

# plt.scatter(y1_test, y1_PR_pred)
# plt.plot(y1_test, y1_test)
# plt.show()



