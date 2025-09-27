import csv
import numpy as np
from utils import get_state, h0, s0, T0, p0, CP
data = {
    "Tsuc [K]": [],
    "psuc [pa]": [],
    "pdis [pa]": [],
    "Theater [K]": [],
    "Tw_in [K]": [],
    "omega [rpm]": [],
    "mdot [g/s]": [],
    "Pcomb [W]": [],
    "Pheating [W]": [],
    "Pcooling [W]": [],
    "Pmotor [W]": [],
    "Pmech [W]": [],
    "Tdis [K]": [],
    "Ploss": [],
    "Pcomp [W]": [],
    "eff [%]": [],
    "Ex_eff [%]": []
}
a_Theater = []
a_Pr = []
a_pcharged = []
a_omega = []
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\results\data_pcharged_Pr_comp.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        data['Tsuc [K]'].append(float(row['Tsuc [°C]']) + 273.15)
        data['psuc [pa]'].append(float(row['psuc [bar]'])*10**5)
        data['pdis [pa]'].append(float(row['pdis [bar]'])*10**5)
        data['Theater [K]'].append(float(row['Theater [°C]']) + 273.15)
        data['Tw_in [K]'].append(float(row['Tw_in [°C]']) + 273.15)
        data['omega [rpm]'].append(float(row['omega [rpm]']))
        data['mdot [g/s]'].append(float(row['mdot [g/s]']))
        data['Pcomb [W]'].append(float(row['Pcomb [W]']))
        data['Pheating [W]'].append(float(row['Pheating [W]']))
        data['Pcooling [W]'].append(float(row['Pcooling [W]']))
        data['Pmotor [W]'].append(float(row['Pmotor [W]']))
        data['Pmech [W]'].append(float(row['Pmech [W]']))
        data['Tdis [K]'].append(float(row['Tdis [°C]']) + 273.15)
        data['Pcomp [W]'].append(float(row['Pcomp [W]']))
        # data['eff [%]'].append(float(row['eff [%]'])*100)
        data['Ploss'].append(float(row['Ploss [W]']))
    
        mw_dot = 10/60
        Theater = data['Theater [K]'][-1]
        omega = data['omega [rpm]'][-1]
        Twi = data['Tw_in [K]'][-1]
        T2 = data['Tdis [K]'][-1]
        p2 = data['pdis [pa]'][-1]
        mdot = data['mdot [g/s]'][-1] * 1e-3
        state = get_state(CP.PT_INPUTS, p2, T2)
        h2, s2 = state.hmass(), state.smass()

        T1 = data['Tsuc [K]'][-1]
        p1 = data['psuc [pa]'][-1]
        state = get_state(CP.PT_INPUTS, p1, T1)
        h1, s1 = state.hmass(), state.smass()
        # --- Compression power and exergy output ---
        Pcomp = data['Pcomp [W]'][-1]
        Pcool  = data['Pcooling [W]'][-1]
        Pheat  = data['Pheating [W]'][-1]
        Pmech  = data['Pmech [W]'][-1]
        psi1 = (h1 - h0) - T0 * (s1 - s0)
        psi2 = (h2 - h0) - T0 * (s2 - s0)
        Ex_comp = mdot * (psi2 - psi1)

        # --- Water side (cooling) ---
        hwi = CP.PropsSI('H', 'P', p0, 'T', Twi, 'Water')
        swi = CP.PropsSI('S', 'P', p0, 'T', Twi, 'Water')
        hwo = hwi + Pcool / mw_dot
        swo = CP.PropsSI('S', 'P', p0, 'H', hwo, 'Water')

        psi_wi = (hwi - h0) - T0 * (swi - s0)
        psi_wo = (hwo - h0) - T0 * (swo - s0)
        Ex_cooler = mw_dot * (psi_wo - psi_wi)

        # --- Exergy terms ---
        Ex_heater = (1 - T0 / Theater) * Pheat
        X_flow   = mdot * T0 * (s2 - s1)
        X_transfer  = -T0 / Theater * Pheat + T0 / Twi * Pcool
        X_heater    = T0 / Theater * Pheat
        X_cooler    = T0 / Twi * Pcool
        X_total     = Ex_heater + Pmech - Ex_comp - Ex_cooler

        # --- Efficiencies ---
        data['Ex_eff [%]'].append(100 * (Ex_comp + Ex_cooler) / (Ex_heater + Pmech))
        a_Pr.append(p2/p1)
        a_Theater.append(Theater)
        a_omega.append(omega)
        a_pcharged.append(np.sqrt(p1 * p2))
# a_Pr =[s/r for s,r in zip(data['pdis [pa]'], data['psuc [pa]'])] 
