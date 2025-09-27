import numpy as np
from CoolProp.CoolProp import AbstractState
from config import CP
import csv


from data_filling_ss import (
     pressure_exp,
     enthalpy_exp
)
import matplotlib.pyplot as plt

def plot_ph(
    pressure_sim,
    enthalpy_sim,
    index
):
    temperature_range = np.linspace(CP.PropsSI('Ttriple', 'CO2'), CP.PropsSI('Tcrit', 'CO2'), 1000)

    # Initialize lists to hold saturation properties
    pressure_liquid = []
    pressure_vapor = []
    enthalpy_liquid = []
    enthalpy_vapor = []

    # Loop through the temperature range and calculate saturation properties
    for T in temperature_range:
        P = CP.PropsSI('P', 'T', T, 'Q', 0, 'CO2')  # Saturation pressure for liquid
        h_l = CP.PropsSI('H', 'T', T, 'Q', 0, 'CO2')  # Enthalpy for liquid
        h_v = CP.PropsSI('H', 'T', T, 'Q', 1, 'CO2')  # Enthalpy for vapor

        pressure_liquid.append(P)
        pressure_vapor.append(P)
        enthalpy_liquid.append(h_l*10**(-3))
        enthalpy_vapor.append(h_v*10**(-3))

    pressure_liquid = np.array(pressure_liquid) / 1e5
    pressure_vapor = np.array(pressure_vapor) / 1e5


    fig = plt.figure()
    plt.plot(enthalpy_liquid, pressure_liquid, enthalpy_vapor, pressure_vapor, color='k')
    plt.plot(enthalpy_exp[index], pressure_exp[index], marker='o', color='g')
    plt.plot(enthalpy_sim, pressure_sim, marker='o', color='r')
    # plt.set_title(f'Point: {labels[index]}, Text: {int(a_Text[index])}') 
    plt.xlabel('Enthalpy [kJ/kg]')
    plt.ylabel('Pressure [bar]')
    plt.grid(True)

    plt.show()