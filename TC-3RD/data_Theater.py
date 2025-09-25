import csv
import numpy as np
from utils import (
	CP,
	state,

)

a_Pr = np.linspace(1.1, 1.65, 5)
a_Theater = [573.15, 673.15, 773.15]
pcharged = 50

data = {
    "Tsuc [K]": [],
    "psuc [pa]": [],
    "pdis [pa]": [],
    "Theater [K]": [],
    "Tw_in [K]": [],
    "omega [rpm]": [],
}

for i in range(len(a_Theater)):
    Theater = a_Theater[i]
    for j in range(len(a_Pr)):
        Pr = a_Pr[j]
        psuc = 2 * pcharged/(1 + Pr) * 1e5
        pdis = 2 * Pr * pcharged/(1 + Pr) * 1e5

        data['psuc [pa]'].append(psuc)
        data['pdis [pa]'].append(pdis)
        state.update(CP.PQ_INPUTS, psuc, 1)
        data['Tsuc [K]'].append(state.T() + 5)
        data['Theater [K]'].append(Theater)
        data['Tw_in [K]'].append(293.15)
        data['omega [rpm]'].append(180)
