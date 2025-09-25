import numpy as np
import CoolProp.CoolProp as CP
import csv
import matplotlib.pyplot as plt


from cycle_model import(
    a_T22,
    a_T9,
    a_Tevap_out,
    a_T20,
    a_Tw_buffer1_out,
    a_hbuffer1_out,
    a_T1,
    a_T2,
    a_T3,
    a_T4,
    a_T5,
    a_T6,
    a_T14,
    a_T10,
    a_T12,
    a_T13,
    a_T15,
    a_T17,
    a_T18,
    a_T19,
    a_T28,
    a_T21,
    a_pevap,
    a_p25,
    a_pi,
    a_pgc,
    a_h11,
    a_h22,
    a_h8,
    a_h23,
    a_h24,
    a_h1,
    a_h2,
    a_h3,
    a_h4,
    a_h5,
    a_h6,
    a_mf1_dot,
    a_mf_dot,
    a_mw_dot,
    a_mw1_dot,
    a_mw2_dot,
    a_mw_dot_buffer1,
    a_mw_dot_buffer2,
    a_mmpg_dot,
    a_hevap_out,
    a_hgc_out,
    a_Theater,
    a_Pcomb,
    a_Pfhx,
    a_Pcooler1,
    a_Pcooler2,
    a_Pcooler3,
    a_Pgc,
    a_Pevap,
    a_PIHX,
    a_Pbuffer1,
    a_Pbuffer2,
    a_Pbuffer3,
    a_PHXs_t,
    a_Pelec,
    a_Text,
    Pheat_out,
    a_Pout_comp,
    SLE,
    GUE,
    a_Pcomp3,
    Res_HP,
    Res_water,
    Res_HP_min,
    Res_HP_max,
    cpw,
    a_mCH4_dot
)
fluid = 'INCOMP::MPG'  # For Monoethylene Glycol solutions
concentration = 0.3  # Glycol concentration by mass (%)
I_gc = []
I_evap = []
I_buff1 = []
I_buff2 = []
I_buff3 = []
I_lpv = []
I_hpv = []
I_ihx1 = []
I_ihx2 = []
I_ihx = []
I_fhx = []
I_comp1 = []
I_comp2 = []
I_comp3 = []
I_tcs = []
I_ft = []
Eheaters = []
Ecoolers = []
Ebuffers = []
Egc_w = []
Eevap_w = []

Egc_in = []
Egc_out = []
Eihx1_in = []
Eihx1_out = []
Eft_in = []
Elpv_in = []
Eevap_in = []
Eihx2_in = []
Eihx2_out = []
Eft_l_out = []
Eft_v_out = []
Ecomb = []
Ex_fhx_in = []
Ex_fhx_out = []
for i in range(len(a_Text)):
    T0 = a_Text[i]
    p0 = 101325
    h0 = CP.PropsSI('H', 'P', p0, 'T', T0, 'Air')
    s0 = CP.PropsSI('S', 'P', p0, 'T', T0, 'Air')

    # TCs
    Ex_comb = (1 - a_Text[i]/1200) * a_Pcomb[i]
    Pheating = a_Pcomb[i] - a_Pfhx[i]
    Ex_heating = (1- a_Text[i]/a_Theater[i]) * (Pheating)

    s14 = CP.PropsSI('S', 'P', p0, 'T', a_T14[i], 'Water')
    h14 = CP.PropsSI('H', 'P', p0, 'T', a_T14[i], 'Water')

    s17 = CP.PropsSI('S', 'P', p0, 'T', a_T17[i], 'Water')
    h17 = CP.PropsSI('H', 'P', p0, 'T', a_T17[i], 'Water')

    psi14 = (h14 - h0) - T0*(s14 - s0)
    psi17 = (h17 - h0) - T0*(s17 - s0)
    Ex_cooling = a_mw1_dot[i] * (psi17 - psi14)

    s1 = CP.PropsSI('S', 'P', a_pevap[i], 'T', a_T1[i], 'CO2')
    s2 = CP.PropsSI('S', 'P', a_p25[i], 'T', a_T2[i], 'CO2')
    s3 = CP.PropsSI('S', 'P', a_p25[i], 'T', a_T3[i], 'CO2')
    s4 = CP.PropsSI('S', 'P', a_pi[i], 'T', a_T4[i], 'CO2')
    s5 = CP.PropsSI('S', 'P', a_pi[i], 'T', a_T5[i], 'CO2')
    s6 = CP.PropsSI('S', 'P', a_pgc[i], 'T', a_T6[i], 'CO2')

    psi1 = (a_h1[i] - h0) - T0*(s1 - s0)
    psi2 = (a_h2[i] - h0) - T0*(s2 - s0)

    psi3 = (a_h3[i] - h0) - T0*(s3 - s0)
    psi4 = (a_h4[i] - h0) - T0*(s4 - s0)

    psi5 = (a_h5[i] - h0) - T0*(s5 - s0)
    psi6 = (a_h6[i] - h0) - T0*(s6 - s0)

    Ex_flow = a_mf1_dot[i] * (psi2 - psi1) +  a_mf_dot[i] * ((psi4 - psi3) + (psi6 - psi5))

    s19 = CP.PropsSI('S', 'P', p0, 'T', a_T19[i], 'Water')
    h19 = CP.PropsSI('H', 'P', p0, 'T', a_T19[i], 'Water')

    s28 = CP.PropsSI('S', 'P', p0, 'T', a_T28[i], 'Water')
    h28 = CP.PropsSI('H', 'P', p0, 'T', a_T28[i], 'Water')

    psi19 = (h19 - h0) - T0*(s19 - s0)
    psi28 = (h28 - h0) - T0*(s28 - s0)
    
    Ex_fhx = a_mw_dot[i] * (psi28 - psi19)

    I_tcs.append(Ex_comb + a_Pelec[i] - Ex_cooling - Ex_flow - Ex_fhx)

    # Gas Cooler
    s11 = CP.PropsSI('S', 'P', a_pgc[i], 'H', a_h11[i], 'CO2')
    sgc_out= CP.PropsSI('S', 'P', a_pgc[i], 'H', a_hgc_out[i], 'CO2')

    psi11 = (a_h11[i] - h0) - T0*(s11 - s0)
    psigc_out = (a_hgc_out[i] - h0) - T0*(sgc_out - s0)

    s9 = CP.PropsSI('S', 'P', p0, 'T', a_T9[i], 'Water')
    h9 = CP.PropsSI('H', 'P', p0, 'T', a_T9[i], 'Water')

    s18 = CP.PropsSI('S', 'P', p0, 'T', a_T18[i], 'Water')
    h18 = CP.PropsSI('H', 'P', p0, 'T', a_T18[i], 'Water')

    psi9 = (h9 - h0) - T0*(s9 - s0)
    psi18 = (h18 - h0) - T0*(s18 - s0)

    sgc_out= CP.PropsSI('S', 'P', a_pgc[i], 'H', a_hgc_out[i], 'CO2')
    I_gc.append(a_mf_dot[i] * (psi11 - psigc_out) - a_mw2_dot[i] * (psi18 - psi9))

    # Evaporator
    s24 = CP.PropsSI('S', 'P', a_pevap[i], 'H', a_h24[i], 'CO2')
    sevap_out= CP.PropsSI('S', 'P', a_pevap[i], 'H', a_hevap_out[i], 'CO2')

    psi24 = (a_h24[i] - h0) - T0*(s24 - s0)
    psievap_out = (a_hevap_out[i] - h0) - T0*(sevap_out - s0)

    s20 = CP.PropsSI('S', 'P', p0, 'T', a_T20[i], fluid + "[%0.1f]" % concentration)
    h20 = CP.PropsSI('H', 'P', p0, 'T', a_T20[i], fluid + "[%0.1f]" % concentration)

    s21 = CP.PropsSI('S', 'P', p0, 'T', a_T21[i], fluid + "[%0.1f]" % concentration)
    h21 = CP.PropsSI('H', 'P', p0, 'T', a_T21[i], fluid + "[%0.1f]" % concentration)

    psi20 = (h20 - h0) - T0*(s20 - s0)
    psi21 = (h21 - h0) - T0*(s21 - s0)

    I_evap.append(a_mmpg_dot[i] * (psi20 - psi21) - a_mf1_dot[i] * (psievap_out - psi24))
    # IHX
    s23 = CP.PropsSI('S', 'P', a_pgc[i], 'H', a_h23[i], 'CO2')
    s1 = CP.PropsSI('S', 'P', a_pevap[i], 'T', a_T1[i], 'CO2')

    psi23 = (a_h23[i] - h0) - T0*(s23 - s0)

    I_ihx1.append(a_mf_dot[i] * (psigc_out - psi23) - a_mf1_dot[i] * (psi1 - psievap_out))
    I_ihx2.append(a_mf1_dot[i] * (psi1 - psievap_out) - a_mf_dot[i] * (psigc_out - psi23))
    I_ihx.append(max(a_Text[i]*(a_mf1_dot[i]*(s1 - sevap_out) - a_mf_dot[i]*(s23 - sgc_out)), 0))

    # Buffers
    sbuffer1_out = CP.PropsSI('S', 'P', a_p25[i], 'H', a_hbuffer1_out[i], 'CO2')
    psibuffer1_out = (a_hbuffer1_out[i] - h0) - T0*(sbuffer1_out - s0)

    sw_buffer1_out = CP.PropsSI('S', 'P', p0, 'T', a_Tw_buffer1_out[i], 'Water')
    hw_buffer1_out = CP.PropsSI('H', 'P', p0, 'T', a_Tw_buffer1_out[i], 'Water')
    psiw_buffer1_out = (hw_buffer1_out - h0) - T0*(sw_buffer1_out - s0)

    s10 = CP.PropsSI('S', 'P', p0, 'T', a_T10[i], 'Water')
    h10 = CP.PropsSI('H', 'P', p0, 'T', a_T10[i], 'Water')

    s12 = CP.PropsSI('S', 'P', p0, 'T', a_T12[i], 'Water')
    h12 = CP.PropsSI('H', 'P', p0, 'T', a_T12[i], 'Water')

    psi10 = (h10 - h0) - T0*(s10 - s0)
    psi12 = (h12 - h0) - T0*(s12 - s0)

    I_buff1.append(max(a_mf1_dot[i] * (psi2 - psibuffer1_out) - a_mw_dot_buffer1[i] * (psiw_buffer1_out - psi9), 0))
    I_buff2.append(max(a_mf_dot[i] * (psi4 - psi5) - a_mw_dot_buffer2[i] * (psi12 - psi10), 0))
    I_buff3.append(a_Text[i]*a_mf_dot[i]*(s6 - s11))

    I_comp1.append(a_Text[i]*a_mf1_dot[i]*(s2 - s1))
    I_comp2.append(a_Text[i]*a_mf_dot[i]*(s4 - s3))
    I_comp3.append(a_Text[i]*a_mf_dot[i]*(s6 - s5))

    s8 = CP.PropsSI('S', 'P', a_p25[i], 'H', a_h8[i], 'CO2')
    psi8 = (a_h8[i] - h0) - T0*(s8 - s0)
    I_hpv.append(a_mf_dot[i]*(psi23 - psi8))

    sft_l = CP.PropsSI('S', 'P', a_p25[i], 'H', a_h24[i], 'CO2')
    s24 = CP.PropsSI('S', 'P', a_pevap[i], 'H', a_h24[i], 'CO2')
    sft_v = CP.PropsSI('S', 'P', a_p25[i], 'Q', 1, 'CO2')
    hft_v = CP.PropsSI('H', 'P', a_p25[i], 'Q', 1, 'CO2')
    psi_ft_l = (a_h24[i] - h0) - T0*(sft_l - s0)

    psi_ft_v = (hft_v - h0) - T0*(sft_v - s0)

    I_lpv.append(a_mf1_dot[i]*(psi_ft_l - psi24))

    Eheaters.append(Ex_heating)
    Ecomb.append(Ex_comb)
    Ecoolers.append(Ex_cooling)
    Ebuffers.append(a_mw_dot_buffer1[i] * (psiw_buffer1_out - psi9) + a_mw_dot_buffer2[i] * (psi12 - psi10))
    Egc_w.append(a_mw2_dot[i] * (psi18 - psi9))
    Eevap_w.append(np.abs(a_mmpg_dot[i] * (psi20 - psi21)))

    Egc_in.append(a_mf_dot[i] * psi11)
    Egc_out.append(a_mf_dot[i] * psigc_out)
    Eihx1_in.append(a_mf_dot[i] * psigc_out)
    Eihx1_out.append(a_mf_dot[i] * psi23)
    Eft_in.append(a_mf_dot[i] * psi23)
    Eft_l_out.append(a_mf1_dot[i] * psi_ft_l)
    Eft_v_out.append((a_mf_dot[i] - a_mf1_dot[i]) * psi_ft_v)
    Elpv_in.append(a_mf1_dot[i] * psi_ft_l)
    Eevap_in.append(a_mf1_dot[i] * psi24)
    Eihx2_in.append(a_mf1_dot[i] * psi1)
    Eihx2_out.append(a_mf1_dot[i] * psievap_out)

    I_ft.append(Eft_in[-1] - Eft_l_out[i] - Eft_v_out[i])
    s13 = CP.PropsSI('S', 'P', p0, 'T', a_T13[i], 'Air')
    h13 = CP.PropsSI('H', 'P', p0, 'T', a_T13[i], 'Air')

    sfume_out = CP.PropsSI('S', 'P', p0, 'T', 40 + 273.15, 'Air')
    hfume_out = CP.PropsSI('H', 'P', p0, 'T', 40 + 273.15, 'Air')

    psi13 = (h13 - h0) - T0*(s13 - s0)
    psifume_out = (hfume_out - h0) - T0*(sfume_out - s0)

    Ex_fhx_in.append(18.25 * a_mCH4_dot[i] * (psi13 - psifume_out))
    Ex_fhx_out.append(a_mw_dot[i] * (psi28 - psi19))

    I_fhx.append(Ex_fhx_in[-1] - Ex_fhx_out[-1])

import plotly.graph_objects as go

k = 12
# Node labels represent different components of the heat pump and energy losses
node_labels = [
    f"Combustion Exergy Rate: {Ecomb[k]:.0f} W", # 0
    f"Electric Power: {a_Pelec[k]:.0f} W",  # 1
    "TCs<br>buffers", # 2
    "GC",  # 3
    "IHX", # 4
    "HPV",   # 5
    "FT", # 6
    "LPV", # 7
    "EVAP",  # 8
    f"Heat output: {Egc_w[k] + Ecoolers[k] + Ebuffers[k] + Ex_fhx_out[k]:.0f} W", # 9
    # "Exergy Loss TCs", # 10
    # "Exergy Loss GC", # 11
    # "Exergy Loss HPV", # 12
    # "Exergy loss FT", # 14
    # "Exergy loss LPV", # 15
    # "Exergy Loss EVAP", # 16
    # f"Total Exergy Destruction Rate: {I_gc[k] + I_evap[k] + I_hpv[k] + I_lpv[k] + I_tcs[k] + I_buff1[k] + I_buff2[k] + I_buff3[k] + I_fhx[k]:.0f} W", # 10
    f"Total Exergy Destruction Rate: {Ecomb[k] + a_Pelec[k] + Eevap_w[k] - (Egc_w[k] + Ecoolers[k] + Ebuffers[k] + Ex_fhx_out[k]):.0f} W", # 10
    f"Environment Exergy Rate: {Eevap_w[k]:.0f} W", # 11
    "FHX", # 12

]

# Energy flow between components, using indices from node_labels
link_source = [0, 1, 6, 2, 2, 2,
                3, 3, 3,
                  4, 4, 5, 5, 6, 6, 7, 7,
                    11, 8, 8, 4, 2, 12, 12]  # Starting points for each flow
 
link_target = [2, 2, 2, 9, 10, 3,
                4, 9, 10,
                  5, 10, 6, 10, 7, 10, 10, 8,
                    8, 10, 4, 2, 12, 9, 10]  # Ending points for each flo


link_value = [Ecomb[k], a_Pelec[k], Eft_in[k] - Elpv_in[k], Ecoolers[k], I_tcs[k], Egc_in[k],
            Egc_out[k], Egc_w[k], I_gc[k],
            Eihx1_out[k], I_ihx[k], Eft_in[k], I_hpv[k], Elpv_in[k], I_ft[k], I_lpv[k], Eevap_in[k],
              Eevap_w[k], I_evap[k], Eihx2_in[k], Eihx2_out[k], Ex_fhx_in[k], Ex_fhx_out[k], I_fhx[k]]  # Energy values for each flow
# Define link colors based on the type of flow or component
opacity = 0.6  # You can change the opacity here
link_colors = [
    f'rgba(200, 130, 0, {opacity})',  # Darker Orange
    f'rgba(120, 120, 0, {opacity})',  # Darker Yellow
    f'rgba(0, 100, 0, {opacity})',    # Darker Green
    f'rgba(240, 0, 0, {opacity})',    # Darker Red
    f'rgba(120, 120, 120, {opacity})',   # Darker Gray
    f'rgba(0, 100, 0, {opacity})',    # Darker Green
    f'rgba(0, 100, 0, {opacity})',    # Darker Green
    f'rgba(240, 0, 0, {opacity})',    # Darker Red
    f'rgba(120, 120, 120, {opacity})',   # Darker Gray
    f'rgba(0, 100, 0, {opacity})',    # Darker Green
    f'rgba(120, 120, 120, {opacity})',   # Darker Gray
    f'rgba(0, 100, 0, {opacity})',    # Darker Green
    f'rgba(120, 120, 120, {opacity})',   # Darker Gray
    f'rgba(0, 100, 0, {opacity})',    # Darker Green
    f'rgba(120, 120, 120, {opacity})',   # Darker Gray
    f'rgba(120, 120, 120, {opacity})',    # Darker Gray
    f'rgba(0, 100, 0, {opacity})',    # Darker Green
    f'rgba(0, 0, 0, {opacity})',    # Darker Gray
    f'rgba(0, 0, 0, {opacity})',    # Darker Gray

    f'rgba(0, 100, 0, {opacity})',    # Darker Green
    f'rgba(0, 100, 0, {opacity})',    # Darker Green
    f'rgba(200, 130, 0, {opacity})',  # Darker Orange
    f'rgba(240, 0, 0, {opacity})',    # Darker Red
    f'rgba(120, 120, 120, {opacity})',   # Darker Gray

]

fig = go.Figure(go.Sankey(
    arrangement='snap',
    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=node_labels, color=['blue'] * len(node_labels)),
    link=dict(source=link_source, target=link_target, value=link_value, color=link_colors)
))
fig.update_layout(title_text="Energy and Exergy Flow in a Heat Pump Cycle", font=dict(size = 18, color = 'black'))
fig.show()