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
    a_T15,
    a_T17,
    a_T13,
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
    Res_TCs,
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
I_ft = []
I_comp1 = []
I_comp2 = []
I_comp3 = []
I_tcs = []
Ex_eff = []

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
COP_thermal = []

for i in range(len(a_Text)):
    T0 = a_Text[i]
    p0 = 101325
    h0 = CP.PropsSI('H', 'P', p0, 'T', T0, 'Air')
    s0 = CP.PropsSI('S', 'P', p0, 'T', T0, 'Air')

    # TCs
    Pheating = a_Pcomb[i] - a_Pfhx[i]
    Ex_heating = (1- a_Text[i]/a_Theater[i]) * (Pheating)
    Tgas = (a_Pcomb[i])/(18.25 * a_mCH4_dot[i] * 1200) + 300 #a_T13[i]

    Ex_comb = (1 - a_Text[i]/Tgas) * a_Pcomb[i]
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
    I_gc.append(max(a_mf_dot[i] * (psi11 - psigc_out) - a_mw2_dot[i] * (psi18 - psi9), 0))

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
    I_ihx2.append(np.abs(a_mf1_dot[i] * (psi1 - psievap_out) - a_mf_dot[i] * (psigc_out - psi23)))
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

    I_buff1.append(a_mf1_dot[i] * (psi2 - psibuffer1_out) - a_mw_dot_buffer1[i] * (psiw_buffer1_out - psi9))
    I_buff2.append(a_mf_dot[i] * (psi4 - psi5) - a_mw_dot_buffer2[i] * (psi12 - psi10))
    I_buff3.append(a_mf_dot[i]*(psi11 - psi6))

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

    I_fhx.append(max(Ex_fhx_in[-1] - Ex_fhx_out[-1], 0))

    Ex_eff.append((a_mw_dot[i] * (psi28 - psi9))/(Ex_comb + a_Pelec[i])) #(Ex_cooling + a_mw2_dot[i] * (psi18 - psi9) + a_mw_dot_buffer2[i] * (psi12 - psi10) + a_mw_dot_buffer1[i] * (psiw_buffer1_out - psi9) + Ex_fhx)/(Ex_comb + a_Pelec[i]))
    COP_thermal.append((a_mw_dot[i] * 4186*  (a_T28[i] - a_T9[i]))/(a_Pcomb[i]))
    print(Ex_comb + a_Pelec[i])

# print(I_gc)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
a_Text = [x - 273.15 for x in a_Text]
# Style settings for publication
mpl.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'figure.figsize': (8, 6),
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'axes.grid': False,
    'grid.alpha': 0.1
})

# Convert to numpy arrays just in case
x = np.arange(len(a_Text))  # Position on x-axis (by outdoor temp index)
bar_width = 0.8
# Stack component values: make sure everything is same length
X_lpv = np.round(np.array(I_lpv) / 1000, 2)
X_hpv = np.round(np.array(I_hpv) / 1000, 2)
X_tc = np.round(np.array(I_tcs) / 1000, 2)
X_ihx = np.round(np.array(I_ihx1) / 1000, 2)
X_buff = np.round(np.array([x + y + z for x, y, z in zip(I_buff1, I_buff2, I_buff3)]) / 1000, 2)
X_gc = np.round(np.array(I_gc) / 1000, 2)
X_evap = np.round(np.array(I_evap) / 1000, 2)
X_ft = np.round(np.array(I_ft) / 1000, 2)
X_fhx = np.round(np.array(I_fhx) / 1000, 2)

# Power vectors
a_Pcomb = np.round(np.array(a_Pcomb) / 1000, 2)
a_Pgc = np.round(np.array(a_Pgc) / 1000, 2)
a_PHXs_t = np.round(np.array(a_PHXs_t) / 1000, 2)
Pheat_out = np.round(np.array(Pheat_out) / 1000, 2)


# Labels and colors (use custom from your plot before)
component_labels = ['LPV', 'HPV', 'TCs', 'IHX', 'Buffers', 'GC', 'EVAP', 'FT', 'FHX']
component_data = [X_lpv, X_hpv, X_tc, X_ihx, X_buff, X_gc, X_evap, X_ft, X_fhx]
component_colors = [
    'gray',         # LPV
    '#BDC3C7',      # HPV
    '#E67E22',      # TCs
    '#F5B041',      # IHX
    '#9B59B6',      # Buffers (keep purple)
    '#3498DB',      # GC
    '#E74C3C',      # EVAP
    '#1ABC9C',      # FT
    '#2ECC71'       # FHX (changed from #8E44AD to #2ECC71, a vibrant green)
]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

x35 = np.arange(9)
x55 = np.arange(9)

bottoms35 = np.zeros_like(x35, dtype=np.float64)
bottoms55 = np.zeros_like(x55, dtype=np.float64)

# W35 subplot
for label, data, color in zip(component_labels, component_data, component_colors):
    ax1.bar(x35, data[:9][::-1], bottom=bottoms35, width=0.8,
            label=label, color=color, edgecolor='black')
    bottoms35 += data[:9][::-1]

ax1.set_xlabel('Outdoor Temperature [°C]')
ax1.set_ylabel('Exergy Destruction [kW]')
ax1.set_xticks(x35)
ax1.set_xticklabels([f"{int(round(t))}" for t in a_Text[:9][::-1]])
# ax1.grid(True, linestyle='--')
ax1.set_title("W35")
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

# Add efficiency line on twin axis
ax1b = ax1.twinx()
ax1b.plot(x35, Ex_eff[:9][::-1], 'k--o', label=r'$\eta_{TCHP, ex}$', linewidth=2, markersize=6)
# ax1b.set_ylabel('Exergy Efficiency [%]', color='k')
ax1b.tick_params(axis='y', labelcolor='k')

# W55 subplot
for label, data, color in zip(component_labels, component_data, component_colors):
    ax2.bar(x55, data[9:][::-1], bottom=bottoms55, width=0.8,
            label=label, color=color, edgecolor='black')
    bottoms55 += data[9:][::-1]

ax2.set_xlabel('Outdoor Temperature [°C]')
ax2.set_xticks(x55)
ax2.set_xticklabels([f"{int(round(t))}" for t in a_Text[9:][::-1]])
# ax2.grid(True, linestyle='--')
ax2.set_title("W55")

# Efficiency line
ax2b = ax2.twinx()
ax2b.plot(x55, Ex_eff[9:][::-1], 'k--o', label=r'$\eta_{ex}$', linewidth=2, markersize=6)
ax2b.set_ylabel('Exergy Efficiency [%]', color='k')
ax2b.tick_params(axis='y', labelcolor='k')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
fig.legend(
    handles=lines1 + lines2,
    labels=labels1 + labels2,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.95),
    ncol=5,
    frameon=True
)

plt.subplots_adjust(top=0.75)
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\W35_W55_exergy_combined.eps",
            format='eps', bbox_inches='tight')


# --- Power Distribution for W35 & W55 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# W35 subplot
line1 = ax1.plot(a_Text[:9], a_Pcomb[:9], 'o-', label='Combustion power', color='tab:red')
line2 = ax1.plot(a_Text[:9], a_Pgc[:9], 's-', label='Gas cooler power', color='tab:blue')
line3 = ax1.plot(a_Text[:9], a_PHXs_t[:9], 'd-', label='Recovered heat', color='tab:green')
line4 = ax1.plot(a_Text[:9], Pheat_out[:9], '^-', label='Total heat output', color='black')

ax1.set_xlabel('Outdoor Temperature [°C]')
ax1.set_ylabel('Power [kW]')
ax1.set_title('W35')
ax1.legend(loc='best', frameon=True)
# ax1.grid(True, linestyle='--')

# Efficiency line (W35)
ax1b = ax1.twinx()
line5 = ax1b.plot(a_Text[:9], COP_thermal[:9], 'o--', color='orange', label='Thermal COP (W35)', linewidth=2, markersize=6)
ax1b.legend(loc='best', frameon=True)
# W55 subplot
line6 = ax2.plot(a_Text[9:], a_Pcomb[9:], 'o-', label='Combustion power', color='tab:red')
line7 = ax2.plot(a_Text[9:], a_Pgc[9:], 's-', label='Gas cooler power', color='tab:blue')
line8 = ax2.plot(a_Text[9:], a_PHXs_t[9:], 'd-', label='Recovered heat', color='tab:green')
line9 = ax2.plot(a_Text[9:], Pheat_out[9:], '^-', label='Total heat output', color='black')

ax2.set_xlabel('Outdoor Temperature [°C]')
ax2.set_title('W55')

# ax2.grid(True, linestyle='--')

# Efficiency line (W55)
ax2b = ax2.twinx()
line10 = ax2b.plot(a_Text[9:], COP_thermal[9:], 'o--', color='orange', label='Thermal COP (W55)', linewidth=2, markersize=6)
ax2b.set_ylabel('Thermal COP [-]', color='k')
ax2b.tick_params(axis='y', labelcolor='k')

# Combine all legend entries
handles = (
    line1 + line2 + line3 + line4 + line5 + line10
)
labels = [h.get_label() for h in handles]


plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\W35_W55_power_distribution.eps", format='eps', bbox_inches='tight')

import matplotlib.pyplot as plt

# Assuming a_Text, Ex_eff, COP_thermal are all 18 elements long
a_Text_W35 = a_Text[:9]
a_Text_W55 = a_Text[9:]

Ex_eff_W35 = Ex_eff[:9]
Ex_eff_W55 = Ex_eff[9:]

COP_thermal_W35 = COP_thermal[:9]
COP_thermal_W55 = COP_thermal[9:]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Exergy Efficiency on left y-axis
color1 = 'tab:blue'
ax1.plot(a_Text_W35, Ex_eff_W35, 'o-', color=color1, label='Exergy Eff. W35')
ax1.plot(a_Text_W55, Ex_eff_W55, 'o--', color=color1, label='Exergy Eff. W55')
ax1.set_xlabel('Outdoor Temperature [°C]', fontsize=16)
ax1.set_ylabel('Exergy Efficiency [-]', color=color1, fontsize=16)
ax1.tick_params(axis='y', labelcolor=color1, labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.grid(True, linestyle='--', alpha=0.4)

# Plot Thermal COP on right y-axis
ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.plot(a_Text_W35, COP_thermal_W35, 's-', color=color2, label='COP W35')
ax2.plot(a_Text_W55, COP_thermal_W55, 's--', color=color2, label='COP W55')
ax2.set_ylabel('Thermal COP [-]', color=color2, fontsize=16)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=14)

# Top axis showing transition
ax_top = ax1.twiny()
ax_top.set_xlim(ax1.get_xlim())
ax_top.set_xticks([a_Text[0], a_Text[8]])
ax_top.set_xticklabels(['Heat Pump', 'Gas Boiler'], fontsize=14)
ax_top.tick_params(axis='x', length=0)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines1 + lines2, labels1 + labels2,
           loc='center left', bbox_to_anchor=(0.1, 0.7),
           borderaxespad=0., fontsize=12)

# plt.title("TCHP Behavior Across Outdoor Temperatures (W35 vs W55)", fontsize=15)
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_GB_or_HP.eps", format='eps', bbox_inches='tight')

plt.show()
