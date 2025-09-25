import csv
import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
# Create an AbstractState object using the HEOS backend and CO2

data = {
    "omegab [rpm]": [],
    "Tw_in [K]": [],

    "T1_in [K]": [],
    "p1_in [pa]": [],
    "p1_out [pa]": [],
    "Theater1 [K]": [],
    "omega1 [rpm]": [],
    "mdot1 [g/s]": [],
    "Pcooling1 [W]": [],
    "T1_out [K]": [],

    "T2_in [K]": [],
    "p2_in [pa]": [],
    "p2_out [pa]": [],
    "Theater2 [K]": [],
    "omega2 [rpm]": [],
    "mdot [g/s]": [],
    "Pcooling2 [W]": [],
    "T2_out [K]": [],

    "T3_in [K]": [],
    "p3_in [pa]": [],
    "p3_out [pa]": [],
    "omega3 [rpm]": [],
    "Pcooling3 [W]": [],
    "T3_out [K]": [],

    "PRt": [],
    "Pmotor [W]": [],
    "Pcomb [W]": [],
    "Pheating [W]": [],
    "Prec_total [W]": [],
    "mw_dot [kg/s]": []
}
counter = 0
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\ML correlations\comps 2.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        counter += 1
        data['omegab [rpm]'].append(float(row['omegab [rpm]']))
        data['Tw_in [K]'].append(float(row['Tw_in [°C]']) + 273.15)
        data['T1_in [K]'].append(float(row['T1_in [°C]']) + 273.15)
        data['p1_in [pa]'].append(float(row['p1_in [bar]'])*10**5)
        data['p1_out [pa]'].append(float(row['p1_out [bar]'])*10**5)
        data['Theater1 [K]'].append(float(row['Theater1 [°C]']) + 273.15)
        data['omega1 [rpm]'].append(float(row['omega1 [rpm]']))
        data['mdot1 [g/s]'].append(float(row['mdot1 [g/s]']))
        data['Pcooling1 [W]'].append(float(row['Pcooling1 [W]']))
        data['T1_out [K]'].append(float(row['T1_out [°C]']) + 273.15)

        data['T2_in [K]'].append(float(row['T2_in [°C]']) + 273.15)
        data['p2_in [pa]'].append(float(row['p2_in [bar]'])*10**5)
        data['p2_out [pa]'].append(float(row['p2_out [bar]'])*10**5)
        data['Theater2 [K]'].append(float(row['Theater2 [°C]']) + 273.15)
        data['omega2 [rpm]'].append(float(row['omega2 [rpm]']))
        data['mdot [g/s]'].append(float(row['mdot [g/s]']))
        data['Pcooling2 [W]'].append(float(row['Pcooling2 [W]']))
        data['T2_out [K]'].append(float(row['T2_out [°C]']) + 273.15)

        data['T3_in [K]'].append(float(row['T3_in [°C]']) + 273.15)
        data['p3_in [pa]'].append(float(row['p3_in [bar]'])*10**5)
        data['p3_out [pa]'].append(float(row['p3_out [bar]'])*10**5)
        data['omega3 [rpm]'].append(float(row['omega3 [rpm]']))
        data['Pcooling3 [W]'].append(float(row['Pcooling3 [W]']))
        data['T3_out [K]'].append(float(row['T3_out2 [°C]']) + 273.15)

        data['PRt'].append(float(row['PRt']))
        data['Pcomb [W]'].append(float(row['Pcomb [W]']))
        data['Prec_total [W]'].append(float(row['Prec_total [W]']))
        data['mw_dot [kg/s]'].append(float(row['mw_dot [l/min]']))


Pr1 = np.array([s/r for s,r in zip(data['p1_out [pa]'][:], data['p1_in [pa]'][:])])
Pr2 = np.array([s/r for s,r in zip(data['p2_out [pa]'][:], data['p2_in [pa]'][:])])
Pr3 = np.array([s/r for s,r in zip(data['p3_out [pa]'][:], data['p3_in [pa]'][:])])

Tr1 = np.array([s/r for s,r in zip(data['Theater1 [K]'][:], data['Tw_in [K]'][:])])
Tr2 = np.array([s/r for s,r in zip(data['Theater2 [K]'][:], data['Tw_in [K]'][:])])

w1 = np.array(data['omega1 [rpm]'])
w2 = np.array(data['omega2 [rpm]'])
w3 = np.array(data['omega3 [rpm]'])

# mCH4_dot = np.array(data['mCH4_dot [kg/s]'])
omegab = np.array(data['omegab [rpm]'])

X1_array = np.array([Pr1, data['Theater1 [K]'], data['Tw_in [K]'], w1])
X2_array = np.array([Pr2, data['Theater2 [K]'], data['Tw_in [K]'], w2])
X3_array = np.array([Pr3, data['Theater2 [K]'], data['Tw_in [K]']])

X4_array = np.array([data['PRt'], omegab, data['Tw_in [K]']])


X1 = X1_array.T
X2 = X2_array.T
X3 = X3_array.T
X4 = X4_array.T


y1_1 = np.array(data['mdot1 [g/s]'])
y1_2 = np.array(data['T1_out [K]'])
y1_3 = np.array(data['Pcooling1 [W]'])

y2_1 = np.array(data['mdot [g/s]'])
y2_2 = np.array(data['T2_out [K]'])
y2_3 = np.array(data['Pcooling2 [W]'])

y3_1 = np.array(data['mdot [g/s]'])
y3_2 = np.array(data['T3_out [K]'])
y3_3 = np.array(data['Pcooling3 [W]'])

y4 = np.array(data['Prec_total [W]'])

from sklearn.model_selection import train_test_split
X1_1_train, X1_1_test, y1_1_train, y1_1_test = train_test_split(X1, y1_1, test_size = 0.01, random_state = 0)
X1_2_train, X1_2_test, y1_2_train, y1_2_test = train_test_split(X1, y1_2, test_size = 0.01, random_state = 0)
X1_3_train, X1_3_test, y1_3_train, y1_3_test = train_test_split(X1, y1_3, test_size = 0.01, random_state = 0)

X2_1_train, X2_1_test, y2_1_train, y2_1_test = train_test_split(X2, y2_1, test_size = 0.01, random_state = 0)
X2_2_train, X2_2_test, y2_2_train, y2_2_test = train_test_split(X2, y2_2, test_size = 0.01, random_state = 0)
X2_3_train, X2_3_test, y2_3_train, y2_3_test = train_test_split(X2, y2_3, test_size = 0.01, random_state = 0)

X3_1_train, X3_1_test, y3_1_train, y3_1_test = train_test_split(X3, y3_1, test_size = 0.01, random_state = 0)
X3_2_train, X3_2_test, y3_2_train, y3_2_test = train_test_split(X3, y3_2, test_size = 0.01, random_state = 0)
X3_3_train, X3_3_test, y3_3_train, y3_3_test = train_test_split(X3, y3_3, test_size = 0.01, random_state = 0)

X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size = 0.01, random_state = 0)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline

poly_degree = 2

poly_model1_1 = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
poly_model1_1.fit(X1_1_train, y1_1_train)

poly_model1_2 = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
poly_model1_2.fit(X1_2_train, y1_2_train)

poly_model1_3 = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
poly_model1_3.fit(X1_3_train, y1_3_train)

poly_model2_1 = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
poly_model2_1.fit(X2_1_train, y2_1_train)

poly_model2_2 = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
poly_model2_2.fit(X2_2_train, y2_2_train)

poly_model2_3 = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
poly_model2_3.fit(X2_3_train, y2_3_train)

poly_model3_1 = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
poly_model3_1.fit(X3_1_train, y3_1_train)

poly_model3_2 = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
poly_model3_2.fit(X3_2_train, y3_2_train)

poly_model3_3 = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
poly_model3_3.fit(X3_3_train, y3_3_train)


poly_model4 = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
poly_model4.fit(X4_train, y4_train)


y1_1_poly_pred = poly_model1_1.predict(X1)
y1_2_poly_pred = poly_model1_2.predict(X1)
y1_3_poly_pred = poly_model1_3.predict(X1)

y2_1_poly_pred = poly_model2_1.predict(X2)
y2_2_poly_pred = poly_model2_2.predict(X2)
y2_3_poly_pred = poly_model2_3.predict(X2)

y3_1_poly_pred = poly_model3_1.predict(X3)
y3_2_poly_pred = poly_model3_2.predict(X3)
y3_3_poly_pred = poly_model3_3.predict(X3)

y4_poly_pred = poly_model4.predict(X4)


import matplotlib.pyplot as plt

# Define a function to create parity plots
def parity_plot_with_metrics(y_true, y_pred, variable_name):
    # Calculate MAPE and R2
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    # Define deviation lines
    y_min, y_max = min(y_true), max(y_true)
    deviation_upper = [1.1 * val for val in y_true]
    deviation_lower = [0.9 * val for val in y_true]

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color='blue', label='Predicted vs Measured')
    plt.plot([y_min, y_max], [y_min, y_max], color='red', linestyle='--', label='Ideal')
    plt.plot(y_true, deviation_upper, color='gray', linestyle='--', label='10% Deviation')
    plt.plot(y_true, deviation_lower, color='gray', linestyle='--')

    # Add text for MAPE and R2
    plt.text(0.05, 0.95, f"MAPE: {mape:.2f}%", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color="darkblue")
    plt.text(0.05, 0.90, f"R²: {r2:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color="darkblue")

    # Plot labels and title
    plt.xlabel(f"Measured {variable_name}")
    plt.ylabel(f"Predicted {variable_name}")
    plt.title(f"Parity Plot for {variable_name}")
    plt.legend()
    plt.grid()
    plt.show()

# Generate parity plots with metrics for all variables
parity_plot_with_metrics(y1_1, y1_1_poly_pred, "Mass Flow Rate 1 (g/s)")
parity_plot_with_metrics(y1_2, y1_2_poly_pred, "Outlet Temperature 1 (K)")
parity_plot_with_metrics(y1_3, y1_3_poly_pred, "Cooling Power 1 (W)")

parity_plot_with_metrics(y2_1, y2_1_poly_pred, "Mass Flow Rate 2 (g/s)")
parity_plot_with_metrics(y2_2, y2_2_poly_pred, "Outlet Temperature 2 (K)")
parity_plot_with_metrics(y2_3, y2_3_poly_pred, "Cooling Power 2 (W)")

parity_plot_with_metrics(y3_1, y3_1_poly_pred, "Mass Flow Rate 3 (g/s)")
parity_plot_with_metrics(y3_2, y3_2_poly_pred, "Outlet Temperature 3 (K)")
parity_plot_with_metrics(y3_3, y3_3_poly_pred, "Cooling Power 3 (W)")

parity_plot_with_metrics(y4, y4_poly_pred, "Total Recovery Power (W)")