import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import csv
import matplotlib.pyplot as plt

from cycle_model import(
    a_T11,
    a_T22,
    a_T9,
    a_T18,
    a_T24,
    a_Tevap_out,
    a_T20,
    a_T21,
    a_T23,
    a_T1,
    a_T2,
    a_T4,
    a_T5,
    a_T10,
    a_T12,
    a_Tbuffer1_out,
    a_Tw_buffer1_out,
    a_Text,
)

a_Tgc = np.zeros((18, 2))
a_Tgc_w = np.zeros((18, 2))
a_Tevap = np.zeros((18, 2))
a_Tevap_w = np.zeros((18, 2))
a_TIHX_1 = np.zeros((18, 2))
a_TIHX_2 = np.zeros((18, 2))
a_Tbuffer1 = np.zeros((18, 2))
a_Tbuffer1_w = np.zeros((18, 2))
a_Tbuffer2 = np.zeros((18, 2))
a_Tbuffer2_w = np.zeros((18, 2))

for i in range(len(a_Text)):
        a_Tgc[i][:] = [a_T11[i]-273.15, a_T22[i]-273.15]
        a_Tgc_w[i][:] = [a_T18[i]-273.15, a_T9[i]-273.15]
        a_Tevap[i][:] = [a_T24[i]-273.15, a_Tevap_out[i]-273.15]
        a_Tevap_w[i][:] = [a_T21[i]-273.15, a_T20[i]-273.15]

        a_TIHX_1[i][:] = [a_T22[i]-273.15, a_T23[i]-273.15]
        a_TIHX_2[i][:] = [a_T1[i]-273.15, a_Tevap_out[i]-273.15]
        a_Tbuffer2[i][:] = [a_T4[i]-273.15, a_T5[i]-273.15]
        a_Tbuffer2_w[i][:] = [a_T10[i]-273.15, a_T12[i]-273.15]
        a_Tbuffer1[i][:] = [a_T2[i]-273.15, a_Tbuffer1_out[i]-273.15]
        a_Tbuffer1_w[i][:] = [a_T9[i]-273.15, a_Tw_buffer1_out[i]-273.15]
        
labels = ['D', 'G', 'C', 'H', 'B', 'F', 'I', 'A', 'Z']
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']

fig1, axes1 = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), sharex=True)
fig1.tight_layout(pad=4.0)
fig1.suptitle('Gas cooler', fontsize=16, y=0.9999)

for index in range(9):
    ax = axes1[index // 3, index % 3]
    ax.plot([1, 2], a_Tgc[index], marker='o', color= colors[index], label = 'refrigerant')
    ax.plot([1, 2], a_Tgc_w[index], linestyle = '--', marker='^', color= colors[index], label = 'fluid')
    ax.set_title(f'Point: {labels[index]}, Text: {int(a_Text[index])}')  
    ax.set_xlabel('point')
    ax.set_ylabel('Temperature [°C]')
    if index == 0:
        ax.legend()
    ax.grid(True)

fig2, axes2 = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), sharex=True)
fig2.tight_layout(pad=4.0)
fig2.suptitle('Evaporator', fontsize=16, y=0.9999)
for index in range(9):
    ax2 = axes2[index // 3, index % 3]
    ax2.plot([1, 2], a_Tevap[index], marker='o', color= colors[index], label = 'refrigerant')
    ax2.plot([1, 2], a_Tevap_w[index], linestyle = '--', marker='^', color= colors[index], label = 'fluid')
    ax2.set_title(f'Point: {labels[index]}, Text: {int(a_Text[index])}')  
    ax2.set_xlabel('point')
    ax2.set_ylabel('Temperature [°C]')
    if index == 0:
        ax2.legend()
    ax2.grid(True)

fig3, axes3 = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), sharex=True)
fig3.tight_layout(pad=4.0)
fig3.suptitle('Internal heat exchanger', fontsize=16, y=0.9999)
for index in range(9):
    ax3 = axes3[index // 3, index % 3]
    ax3.plot([1, 2], a_TIHX_1[index], marker='o', color= colors[index], label = 'refrigerant at path 1')
    ax3.plot([1, 2], a_TIHX_2[index], linestyle = '--', marker='^', color= colors[index], label = 'refrigerant at path 2')
    ax3.set_title(f'Point: {labels[index]}, Text: {int(a_Text[index])}')  
    ax3.set_xlabel('point')
    ax3.set_ylabel('Temperature [°C]')
    if index == 0:
        ax3.legend()
    ax3.grid(True)

fig4, axes4 = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), sharex=True)
fig4.tight_layout(pad=4.0)
fig4.suptitle('Buffer 1', fontsize=16, y=0.9999)
for index in range(9):
    ax4 = axes4[index // 3, index % 3]
    ax4.plot([1, 2], a_Tbuffer1[index], marker='o', color= colors[index], label = 'refrigerant')
    ax4.plot([1, 2], a_Tbuffer1_w[index], linestyle = '--', marker='^', color= colors[index], label = 'fluid')
    ax4.set_title(f'Point: {labels[index]}, Text: {int(a_Text[index])}')  
    ax4.set_xlabel('point')
    ax4.set_ylabel('Temperature [°C]')
    if index == 0:
        ax4.legend()
    ax4.grid(True)

fig5, axes5 = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), sharex=True)
fig5.tight_layout(pad=4.0)
fig5.suptitle('Buffer 2', fontsize=16, y=0.9999)
for index in range(9):
    ax5 = axes5[index // 3, index % 3]
    ax5.plot([1, 2], a_Tbuffer2[index], marker='o', color= colors[index], label = 'refrigerant')
    ax5.plot([1, 2], a_Tbuffer2_w[index], linestyle = '--', marker='^', color= colors[index], label = 'fluid')
    ax5.set_title(f'Point: {labels[index]}, Text: {int(a_Text[index])}')  
    ax5.set_xlabel('point')
    ax5.set_ylabel('Temperature [°C]')
    if index == 0:
        ax5.legend()
    ax5.grid(True)

plt.show()
# fig1 = plt.figure()
# plt.plot(a_LMTD_gc, a_Pgc,  marker='o')
# plt.xlabel('LMTD', fontsize = 16)
# plt.ylabel('Gas cooler power [kW]', fontsize = 16) 
# plt.xticks(fontsize = 14)
# plt.yticks(fontsize = 14)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()

# fig2 = plt.figure()
# plt.plot(a_LMTD_evap, a_Pevap,  marker='o')
# plt.xlabel('LMTD', fontsize = 16)
# plt.ylabel('Evaporator power [kW]', fontsize = 16) 
# plt.xticks(fontsize = 14)
# plt.yticks(fontsize = 14)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()

# fig3 = plt.figure()
# plt.plot(a_LMTD_IHX, a_PIHX,  marker='o')
# plt.xlabel('LMTD', fontsize = 16)
# plt.ylabel('IHX power [kW]', fontsize = 16) 
# plt.xticks(fontsize = 14)
# plt.yticks(fontsize = 14)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()

# fig4 = plt.figure()
# plt.plot(a_LMTD_buffer1, a_Pbuffer1,  marker='o')
# plt.xlabel('LMTD', fontsize = 16)
# plt.ylabel('bufffer 1 power [kW]', fontsize = 16) 
# plt.xticks(fontsize = 14)
# plt.yticks(fontsize = 14)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()

# fig5 = plt.figure()
# plt.plot(a_LMTD_buffer2, a_Pbuffer2,  marker='o')
# plt.xlabel('LMTD', fontsize = 16)
# plt.ylabel('buffer 2 power [kW]', fontsize = 16) 
# plt.xticks(fontsize = 14)
# plt.yticks(fontsize = 14)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()









