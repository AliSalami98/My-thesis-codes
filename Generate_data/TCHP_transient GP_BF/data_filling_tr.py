import csv
import numpy as np
from utils import (
	get_state,
)
from config import CP
import config
# Initialize lists to hold the column data
d_t = []
d_Hpev = []
d_Lpev = []
d_mw_dot = []
d_mmpg_dot = []
d_Tw_in = []
d_Tmpg_in = []
d_Tw_out = []
d_Tmpg_out = []
d_pc = []
d_p25 = []
d_pi = []
d_pe = []
d_Tc_out = []
d_Te_out = []
d_omega1 = []
d_omega2 = []
d_omega3 = []
d_omegab = []
d_Theater1 = []
d_Theater2 = []
d_Pcomb = []
d_Pheat_out = []
d_Pevap = []
d_COP = []
d_SH = []
d_error = []
i = 0
step_counter = 0

counter_threshold = 60
# Read the CSV file
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\LPV PRBS.csv') as csv_file:
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\HPV PRBS.csv') as csv_file:
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\bf step.csv') as csv_file: #(For this step make sure step_counter is incremented)
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\Tw step.csv') as csv_file:
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\mw step.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        if step_counter % counter_threshold == 0:
            try:
                if row['omegab [rpm]']:
                    d_omegab.append(float(row['omegab [rpm]']))
                if row['Tw_in [°C]']:
                    d_Tw_in.append(float(row['Tw_in [°C]']) + 273.15)
                if row['Tmpg_in [°C]']:
                    d_Tmpg_in.append(float(row['Tmpg_in [°C]']) + 273.15)
                if row['mmpg_dot [l/min]']:
                    d_mmpg_dot.append(float(row['mmpg_dot [l/min]'])/60)
                if row['mw_dot [l/min]']:
                    d_mw_dot.append(float(row['mw_dot [l/min]'])/60)
                if row['Hpev']:
                    d_Hpev.append(float(row['Hpev']))
                if row['Lpev']:
                    d_Lpev.append(float(row['Lpev']))
                if row['omega1 [rpm]']:
                    d_omega1.append(float(row['omega1 [rpm]']))
                if row['omega2 [rpm]']:
                    d_omega2.append(float(row['omega2 [rpm]']))
                if row['omega3 [rpm]']:
                    d_omega3.append(float(row['omega3 [rpm]']))         
                if row['Theater1 [°C]']:
                    d_Theater1.append(float(row['Theater1 [°C]']) + 273.15)
                if row['Theater2 [°C]']:
                    d_Theater2.append(float(row['Theater2 [°C]']) + 273.15)
                if row['pc [bar]']:
                    d_pc.append(float(row['pc [bar]']) * 10**5)
                if row['p25 [bar]']:
                    d_p25.append(float(row['p25 [bar]']) * 10**5)
                if row['pi [bar]']:
                    d_pi.append(float(row['pi [bar]']) * 10**5)
                if row['pe [bar]']:
                    d_pe.append(float(row['pe [bar]']) * 10**5)
                if row['Tc_out [°C]']:
                    d_Tc_out.append(float(row['Tc_out [°C]']) + 273.15)
                if row['Te_out [°C]']:
                    d_Te_out.append(float(row['Te_out [°C]']) + 273.15)
                # if row['Pheat_out [W]']:
                #     d_Pheat_out.append(float(row['Pheat_out [W]']))
                # if row['Pcomb [W]']:
                #     d_Pcomb.append(float(row['Pcomb [W]']))
                # if row['COP']:
                #     d_COP.append(float(row['COP']))
                # if row['Pevap [W]']:
                #     d_Pevap.append(float(row['Pevap [W]']))
                if row['Tw_out [°C]']:
                    d_Tw_out.append(float(row['Tw_out [°C]']) + 273.15)
                if row['Tmpg_out [°C]']:
                    d_Tmpg_out.append(float(row['Tmpg_out [°C]']) + 273.15)
            except ValueError as e:
                    print(f"Skipping row {i} due to error: {e}")


        step_counter += 1         

d_Te_out = np.interp(np.linspace(0, len(d_Te_out) - 1, len(d_omegab)), np.arange(len(d_Te_out)), d_Te_out)
d_Tc_out = np.interp(np.linspace(0, len(d_Tc_out) - 1, len(d_omegab)), np.arange(len(d_Tc_out)), d_Tc_out)

LHV = 50e6
for i in range(len(d_omegab)):
    mCH4_dot = (0.0022 * d_omegab[i] - 2.5965) * 0.657/60000
    d_Pcomb.append(LHV * mCH4_dot)
    d_Pheat_out.append(d_mw_dot[i] * config.cp_w * (d_Tw_out[i] - d_Tw_in[i]))
    d_COP.append(d_Pheat_out[-1]/d_Pcomb[-1])
    state = get_state(CP.PQ_INPUTS, d_pe[i], 1)
    d_SH.append(d_Te_out[i] - state.T())
    d_Pevap.append(d_mmpg_dot[i] * config.cp_mpg * (d_Tmpg_in[i] - d_Tmpg_out[i]))
    d_error.append(d_Pcomb[-1] - d_Pheat_out[-1] + d_Pevap[-1])


t_init = 80

d_omegab = d_omegab[t_init:-1]
d_omega1 = d_omega1[t_init:-1]
d_omega2 = d_omega2[t_init:-1]
d_omega3 = d_omega3[t_init:-1]
d_Theater1 = d_Theater1[t_init:-1]
d_Theater2 = d_Theater2[t_init:-1]
d_mw_dot = d_mw_dot[t_init:-1]
d_mmpg_dot = d_mmpg_dot[t_init:-1]
d_Lpev = d_Lpev[t_init:-1]
d_Hpev = d_Hpev[t_init:-1]
d_Tw_in = d_Tw_in[t_init:-1]
d_Tmpg_in = d_Tmpg_in[t_init:-1]
d_Tc_out = d_Tc_out[t_init:-1]
d_Te_out = d_Te_out[t_init:-1]
d_SH = d_SH[t_init:-1]

d_Tw_out = d_Tw_out[t_init:-1]
d_Tmpg_out = d_Tmpg_out[t_init:-1]
d_pc = d_pc[t_init:-1]
d_p25 = d_p25[t_init:-1]
d_pi = d_pi[t_init:-1]
d_pe = d_pe[t_init:-1]
d_Pevap = d_Pevap[t_init:-1]
d_Pheat_out = d_Pheat_out[t_init:-1]
d_Pcomb = d_Pcomb[t_init:-1]
d_COP = d_COP[t_init:-1]
d_error = d_error[t_init:-1]

