import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
a_T18 = []
a_T24 = []
a_T20 = []
a_T21 = []
a_T23 = []
a_T1 = []
a_T2 = []
a_T3 = []
a_T4 = []
a_T5 = []
a_T6 = []
a_T10 = []
a_T12 = []
a_Tfume = []
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

a_Tevap_out = []

a_Cd = []
a_Delp = []
a_D_gc = []
a_D_IHX = []
a_D_evap = []
a_D_buffer1 = []
a_D_buffer2 = []
a_D_rec = []
a_AU_gc = []
a_AU_IHX = []
a_AU_evap = []
a_AU_buffer1 = []
a_AU_buffer2 = []
a_AU_rec = []
a_LMTD_gc = []
a_LMTD_gc1 = []
a_LMTD_gc3 = []

a_LMTD_IHX = []
a_LMTD_evap = []
a_LMTD_buffer1 = []
a_LMTD_buffer2 = []
a_LMTD_rec = []
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

a_Pgc1 = []
a_Re1 = []
a_Pr1 = []
a_Nu1 = []

a_Pgc2 = []
a_Re2 = []
a_Pr2 = []
a_Nu2 = []

a_Pgc3 = []
a_Re3 = []
a_Pr3 = []
a_Nu3 = []

a_Xc = []
a_Pgcw = []
a_Rew = []
a_Prw = []
a_Nuw = []

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

i = 0
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\cycle measurements\all.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        if i >= 0 :
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
            T28 = float(row['T28'])+ 273.15
            Theater = float(row['Theater1'])+ 273.15
            Tw_return = T9
            Tw_supply = T28 #float(row['Tw_supply'])+ 273.15
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

            Ac_surface = 0.0193 * 54 / 10
            dhc = 12.3 * 10 ** (-3)
            Ac_orifice = np.pi / 4 * (12.3 * 10 ** (-3)) ** 2
            Tgc_wall = (T9 + T18)/2

            D22 = CP.PropsSI('DMASS', 'P', pgc, 'H', hgc_out, 'CO2')
            mu11 = CP.PropsSI('V', 'P', pgc, 'T', T11, 'CO2')
            mu22 = CP.PropsSI('V', 'P', pgc, 'H', hgc_out, 'CO2')

            cp11 = CP.PropsSI('CPMASS', 'P', pgc, 'T', T11, 'CO2')
            cp22 = np.abs(CP.PropsSI('CPMASS', 'P', pgc, 'H', hgc_out, 'CO2'))

            k11 = CP.PropsSI('CONDUCTIVITY', 'P', pgc, 'T', T11, 'CO2')
            k22 = CP.PropsSI('CONDUCTIVITY', 'P', pgc, 'H', hgc_out, 'CO2')

            DTgc_1 = max(T11 - T18, 0.1)
            DTgc_2 =max(T22 - T9, 0.1)
            LMTD_gc = T11 - T18 #(DTgc_1 - DTgc_2)/np.log(DTgc_1/DTgc_2) #(T11 + T22 - (T9 + T18))/2 #

            if 0 < quality_vapor < 1:
                hc_v = CP.PropsSI('HMASS', 'P', pgc, 'Q', 1, 'CO2')
                Dc_v = CP.PropsSI('DMASS', 'P', pgc, 'Q', 1, 'CO2')
                muc_v = CP.PropsSI('V', 'P', pgc, 'Q', 1, 'CO2')
                kc_v = CP.PropsSI('CONDUCTIVITY', 'P', pgc, 'Q', 1, 'CO2')
                cpc_v = CP.PropsSI('CPMASS', 'P', pgc, 'Q', 1, 'CO2')

                Tc_tp = CP.PropsSI('T', 'P', pgc, 'Q', 0, 'CO2')
                Tc1 = T11
                Tc2 = Tc_tp

                Dc1 = (D11 + Dc_v)/2
                Dc2 =  (D22 + Dc_v)/2

                muc1 = (muc_v + mu11)/2
                muc2 = (muc_v + mu22)/2

                kc1 = (kc_v + k11)/2
                kc2 = (kc_v + k22)/2

                cpc1 = (cpc_v + cp11)/2
                cpc2 = (cpc_v + cp22)/2

                vc1 = mf_dot/(Dc1 * Ac_orifice)
                vc2 = mf_dot/(Dc2 * Ac_orifice)

                Re1 = vc1 * dhc * Dc1 / muc1
                Pr1 = muc1 * cpc1 / kc1
                Re2 = vc2 * dhc * Dc2 / muc2
                Pr2 = muc2 * cpc2 / kc2

                # print(cpc_v)

                Pgc1 = mf_dot * (h11 - hc_v)
                Pgc2 = mf_dot * (hc_v - hgc_out)

                Uc1 = Pgc1/(Ac_surface *  LMTD_gc)
                Uc2 = Pgc2/(Ac_surface * LMTD_gc)

                Nuc1 = dhc * Uc1/kc1
                Nuc2 = dhc * Uc2/kc2

                a_Pgc1.append(Pgc1)
                a_Re1.append(Re1)
                a_Pr1.append(Pr1)
                a_Nu1.append(Uc1)
                a_LMTD_gc1.append(LMTD_gc)

                a_Pgc2.append(Pgc2)
                a_Re2.append(Re2)
                a_Pr2.append(Pr2)
                a_Nu2.append(Uc2)

                # a_Pgc2.append(Pgc_w)
                # a_Re2.append((Re1 + Re2)/2)
                # a_Pr2.append((Pr1 + Pr2)/2)
                # a_Nu2.append((Nuc1 + Nuc2)/2)
                a_Xc.append(quality_vapor)

            else:
                Dc3 = (D11 + D22)/2
                muc3 = (mu11 + mu22)/2
                cpc3 = (cp11 + cp22)/2
                kc3 = (k11 + k22)/2
                Tc3 = (T11 + Tgc_out)/2

                vc3 = mf_dot/(Dc3 * Ac_orifice)
                Re3 = vc3 * dhc * Dc3 / muc3
                Pr3 = muc3 * cpc3 / kc3
                Uc3 = Pgc_w/(Ac_surface * LMTD_gc)
                Nuc3 = dhc * Uc3/kc3

                a_Pgc3.append(Pgc_w)
                a_Re3.append(Re3)
                a_Pr3.append(Pr3)
                a_Nu3.append(Uc3)
                a_LMTD_gc3.append(LMTD_gc)


            Aw_orifice = np.pi / 4 * (20 * 10 ** (-3)) ** 2
            dhw = 20 * 10 ** (-3)
            muw = 8.9 * 10 ** (-4)
            kw = 0.6

            vw = mw2_dot/(100*Dw * Aw_orifice)
            Rew = vw * dhw * 1000* Dw / muw
            Prw = muw * cpw / kw
            Uw = Pgc_w/(Ac_surface * (Tc3 - Tgc_wall))
            Nuw = dhw * Uw/kw

            a_Pgcw.append(Pgc_w)
            a_Rew.append(Rew)
            a_Prw.append(Prw)
            a_Nuw.append(Nuw)

            # DTgc_1 = np.abs(T11 - T18)
            # DTgc_2 = np.abs(T22 - T9)
            # LMTD_gc = (DTgc_1 - DTgc_2)/np.log(DTgc_1/DTgc_2) #(T11 + T22 - (T9 + T18))/2 #

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
            a_T10.append(T10)
            a_T12.append(T12)
            a_Tbuffer1_out.append(Tbuffer1_out)
            a_Tw_buffer1_out.append(Tw_buffer1_out)
            a_Theater.append(Theater)
            a_Tfume.append(T13)
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
            a_mf_dot.append(mf_dot)
            a_mf1_dot.append(mf1_dot)
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
            a_mhp_dot.append(Hpev*np.sqrt(2*D23*(pgc - p25)))
            a_mbp_dot.append(Lpev*np.sqrt(2*D24*(p25 - pevap)))
            a_Hpev.append(Hpev)
            a_Lpev.append(Lpev)
            a_DpHp.append(pgc - p25)
            a_DpLp.append(p25 - pevap)

            a_Delp.append(pgc-p25)
            a_Cd.append(mf_dot/a_mhp_dot[-1])
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
            a_PHXs_t.append(Pcooler1 + Pcooler2 + Pcooler3 + Pbuffer1 + Pbuffer2 + Pfhx)
            a_Pelec.append(Pelec)
            Pin_comp = Pcomb
            a_Pin_comp.append(Pin_comp)
            Pout_comp = Pcomp1 + Pcomp2 + Pcomp3 + Pcooler1 + Pcooler2 + Pcooler3 + Pfhx
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
            Res_HP.append((Pcomb + Pevap + Pelec) -  mw_dot*cpw*(Tw_supply - Tw_return))

        i = i+1

print(np.mean(mf_dot))
def dittus_boelter(Re, Pr, Xc, C, a, b, d):
    return C * Re**a * Pr**b *Xc**d

# Combine Re and Pr into a single array
def model(X, C, a, b, d):
    Re, Pr, Xc = X
    return dittus_boelter(Re, Pr, Xc, C, a, b, d)

# Perform the curve fitting
initial_guess = [0.023, 0.8, 0.3, 0.2]  # Initial guess for C, a, b
params, covariance = curve_fit(model, (a_Re2, a_Pr2, a_Xc), a_Nu2, p0=initial_guess)

# Extract the fitted parameters
C_fitted, a_fitted, b_fitted, d_fitted = params

print(f"Fitted parameters: C = {C_fitted}, a = {a_fitted}, b = {b_fitted}, d = {d_fitted}")

# Plot the data vs the fitted model
Nu_fitted = model((a_Re2, a_Pr2, a_Xc), *params)

# def dittus_boelter(Re, Pr, C, a, b):
#     return C * Re**a * Pr**b

# # Combine Re and Pr into a single array
# def model(X, C, a, b):
#     Re, Pr = X
#     return dittus_boelter(Re, Pr, C, a, b)

# # Perform the curve fitting
# initial_guess = [0.023, 0.8, 0.3]  # Initial guess for C, a, b
# params, covariance = curve_fit(model, (a_Re2, a_Pr2), a_Nu2, p0=initial_guess)

# # Extract the fitted parameters
# C_fitted, a_fitted, b_fitted = params

# print(f"Fitted parameters: C = {C_fitted}, a = {a_fitted}, b = {b_fitted}")

# # Plot the data vs the fitted model
# Nu_fitted = model((a_Re2, a_Pr2), *params)
print(np.mean(Nu_fitted))
a_Pgc_tp = [Uc * Ac_surface * LMTD_gc for Uc, LMTD_gc in zip(Nu_fitted, a_LMTD_gc3)]

plt.scatter(a_Pgc3, a_Pgc_tp, label='Fitted Data', color='blue')
plt.plot([min(a_Pgc3), max(a_Pgc3)], [min(a_Pgc3), max(a_Pgc3)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Observed Gas cooler power')
plt.ylabel('Fitted Gas cooler power')
plt.legend()
plt.title('Calibration of Dittus-Boelter Equation')
plt.grid(True)

# plt.scatter(a_Nu3, Nu_fitted, label='Fitted Data', color='blue')
# plt.plot([min(a_Nu3), max(a_Nu3)], [min(a_Nu3), max(a_Nu3)], color='red', linestyle='--', label='Ideal Fit')
# plt.xlabel('Observed Nu')
# plt.ylabel('Fitted Nu')
# plt.legend()
# plt.title('Calibration of Dittus-Boelter Equation')
# plt.grid(True)
plt.show()

