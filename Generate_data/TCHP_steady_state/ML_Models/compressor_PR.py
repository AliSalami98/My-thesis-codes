import csv
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from scipy.optimize import minimize

# Example constraints for mass flow rate (mdot [g/s])
mdot_min = 2  # Minimum allowable mass flow rate (example value)
mdot_max = 40 # Maximum allowable mass flow rate (example value)

# Constrained prediction function for mass flow rate
def constrained_predict(X, model, poly, min_value, max_value):
    """
    Constrain the prediction of a polynomial regression model.

    Parameters:
        X (array-like): Input data for prediction.
        model (sklearn model): Trained polynomial regression model.
        poly (PolynomialFeatures): The polynomial feature transformer.
        min_value (float): Minimum constraint for the prediction.
        max_value (float): Maximum constraint for the prediction.

    Returns:
        float: Constrained prediction.
    """
    # Polynomial transformation of the input
    X_poly = poly.transform(X.reshape(1, -1))

    # Initial unconstrained prediction
    y_pred = model.predict(X_poly)[0]

    # Define the objective function for minimizing the difference with initial prediction
    def objective(y):
        return (y - y_pred) ** 2

    # Define constraints for the optimization
    constraints = [
        {'type': 'ineq', 'fun': lambda y: y - min_value},  # y >= min_value
        {'type': 'ineq', 'fun': lambda y: max_value - y}   # y <= max_value
    ]

    # Optimize to find the closest value within constraints
    result = minimize(objective, y_pred, constraints=constraints, method='SLSQP')

    # Return the constrained prediction
    return result.x[0]


data = {
    "Tin [K]": [],
    "pin [pa]": [],
    "pout [pa]": [],
    "Th_wall [K]": [],
    "Tw_in [K]": [],
    "omega [rpm]": [],
    "mdot [g/s]": [],
    "Pcomb [W]": [],
    "Pheating [W]": [],
    "Pcooling [W]": [],
    "Pmotor [W]": [],
    "Tout [K]": [],
    "Pout [W]": [],
    "eff [%]": [],
}

counter = 0
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\ML correlations\I-O data4.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        counter += 1
        data['Tin [K]'].append(float(row['Tin [°C]']) + 273.15)
        data['pin [pa]'].append(float(row['pin [bar]'])*10**5)
        data['pout [pa]'].append(float(row['pout [bar]'])*10**5)
        data['Th_wall [K]'].append(float(row['Th_wall[°C]']) + 273.15)
        data['Tw_in [K]'].append(float(row['Tw_in [°C]']) + 273.15)
        data['omega [rpm]'].append(float(row['omega [rpm]']))
        data['mdot [g/s]'].append(float(row['mdot [g/s]']))
        # data['Pcomb [W]'].append(float(row['Pcomb [W]']))
        # data['Pheating [W]'].append(float(row['Pheating [W]']))
        data['Pcooling [W]'].append(float(row['Pcooling [W]']))
        # data['Pmotor [W]'].append(float(row['Pmotor [W]']))
        data['Tout [K]'].append(float(row['Tout [°C]']) + 273.15)
        # data['Pout [W]'].append(float(row['Pout [W]']))
        # data['eff [%]'].append(100*float(row['eff [%]']))
        # if counter > 52:
        #     break

Pr = [s/r for s,r in zip(data['pout [pa]'][:], data['pin [pa]'][:])]
Tr = [s/r for s,r in zip(data['Th_wall [K]'][:], data['Tw_in [K]'][:])]

# X1_array = np.array([data['pin [pa]'],data['pout [pa]'], data['omega [rpm]'], data['Th_wall [K]'], data['Tw_in [K]']])
X1_array = np.array([Pr, Tr, data['omega [rpm]']])

X1 = X1_array.T

y1 = np.array(data['mdot [g/s]'])
y2 = np.array(data['Tout [K]'])
y3 = np.array(data['Pcooling [W]'])


from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X1, y2, test_size = 0.2, random_state = 0)
X3_train, X3_test, y3_train, y3_test = train_test_split(X1, y3, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly1_reg = PolynomialFeatures(degree = 2, interaction_only = True)
X1_poly = poly1_reg.fit_transform(X1_train)
model1_PR = LinearRegression()
model1_PR.fit(X1_poly, y1_train)

poly2_reg = PolynomialFeatures(degree = 2, interaction_only = True)
X2_poly = poly2_reg.fit_transform(X2_train)
model2_PR = LinearRegression()
model2_PR.fit(X2_poly, y2_train)

poly3_reg = PolynomialFeatures(degree = 2, interaction_only = True)
X3_poly = poly3_reg.fit_transform(X3_train)
model3_PR = LinearRegression()
model3_PR.fit(X3_poly, y3_train)

y1_PR_pred = model1_PR.predict(poly1_reg.transform(X1_test))
# y1_PR_pred = [constrained_predict(X, model1_PR, poly1_reg, mdot_min, mdot_max) for X in X1_test]

y2_PR_pred = model2_PR.predict(poly2_reg.transform(X2_test))
y3_PR_pred = model3_PR.predict(poly3_reg.transform(X3_test))


mape1_PR = mean_absolute_percentage_error(y1_test, y1_PR_pred) *100
mape2_PR = mean_absolute_percentage_error(y2_test, y2_PR_pred) *100
mape3_PR = mean_absolute_percentage_error(y3_test, y3_PR_pred) *100

Rsquare1_PR = r2_score(y1_test, y1_PR_pred) *100
Rsquare2_PR = r2_score(y2_test, y2_PR_pred) *100
Rsquare3_PR = r2_score(y3_test, y3_PR_pred) *100

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

X1_array = np.array([Pr1, Tr1, w1])
X2_array = np.array([Pr2, Tr2, w2])
X3_array = np.array([Pr3, omegab])

X4_array = np.array([data['PRt'], omegab, data['Tw_in [K]']])

# X1_array = np.array([Pr1, w1, data['Tw_in [K]']])
# X2_array = np.array([Pr2, Tr2, w2])
# X3_array = np.array([Pr3, omegab, data['Tw_in [K]']])

# X1_array = np.array([data['p1_in [pa]'], data['p1_out [pa]'], data['omega1 [rpm]'], data['Tw_in [K]']])
# X2_array = np.array([data['p2_in [pa]'], data['p2_out [pa]'], data['omega2 [rpm]'], omegab])
# X3_array = np.array([data['p3_in [pa]'], data['p3_out [pa]'], data['omega3 [rpm]'], omegab])

# X1_array = np.array([Pr, Tr]) #, data['omega [rpm]']])

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

model1_1_LR = LinearRegression()
model1_1_LR.fit(X1_1_train, y1_1_train)

model1_2_LR = LinearRegression()
model1_2_LR.fit(X1_2_train, y1_2_train)

model1_3_LR = LinearRegression()
model1_3_LR.fit(X1_3_train, y1_3_train)

model2_1_LR = LinearRegression()
model2_1_LR.fit(X2_1_train, y2_1_train)

model2_2_LR = LinearRegression()
model2_2_LR.fit(X2_2_train, y2_2_train)

model2_3_LR = LinearRegression()
model2_3_LR.fit(X2_3_train, y2_3_train)

model3_1_LR = LinearRegression()
model3_1_LR.fit(X3_1_train, y3_1_train)

model3_2_LR = LinearRegression()
model3_2_LR.fit(X3_2_train, y3_2_train)

model3_3_LR = LinearRegression()
model3_3_LR.fit(X3_3_train, y3_3_train)


model4_LR = LinearRegression()
model4_LR.fit(X4_train, y4_train)

y1_1_pred = model1_1_LR.predict(X1)
y1_2_pred = model1_2_LR.predict(X1_2_test)
y1_3_pred = model1_3_LR.predict(X1)

y2_1_pred = model2_1_LR.predict(X2)
y2_2_pred = model2_2_LR.predict(X2_2_test)
y2_3_pred = model2_3_LR.predict(X2)

y3_1_pred = model3_1_LR.predict(X3_1_test)
y3_2_pred = model3_2_LR.predict(X3_2_test)
y3_3_pred = model3_3_LR.predict(X3_3_test)

Pcooler1_PR = model3_PR.predict(poly3_reg.transform(X1))
Pcooler2_PR = model3_PR.predict(poly3_reg.transform(X2))
mdot_PR = model1_PR.predict(poly1_reg.transform(X2))
mdot1_PR = model1_PR.predict(poly1_reg.transform(X1))


# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# y_true = y1_1

# # Compare Metrics
# mae_poly = mean_absolute_error(y_true, mdot1_PR)
# mape_poly = mean_absolute_percentage_error(y_true, mdot1_PR) * 100
# r2_poly = r2_score(y_true, mdot1_PR)

# mae_lr = mean_absolute_error(y_true, y1_1_pred)
# mape_lr = mean_absolute_percentage_error(y_true, y1_1_pred) * 100
# r2_lr = r2_score(y_true, y1_1_pred)

# # Print Comparison Metrics
# print("Comparison of mdot1_PR (Polynomial Regression) and y1_1_pred (Linear Regression):")
# print(f"Polynomial Regression - MAE: {mae_poly:.2f}, MAPE: {mape_poly:.2f}%, R²: {r2_poly:.2f}")
# print(f"Linear Regression      - MAE: {mae_lr:.2f}, MAPE: {mape_lr:.2f}%, R²: {r2_lr:.2f}")

# # Plot Ground Truth vs Predictions
# plt.figure(figsize=(12, 6))

# plt.plot(y_true, label="Ground Truth (y1_1_test)", color="black", linewidth=2)
# plt.plot(mdot1_PR, label="mdot1_PR (Polynomial Regression)", linestyle="--", color="blue")
# plt.plot(y1_1_pred, label="y1_1_pred (Linear Regression)", linestyle="--", color="red")

# plt.xlabel("Test Data Index")
# plt.ylabel("mdot [g/s]")
# plt.title("Comparison of Polynomial and Linear Regression Predictions")
# plt.legend()
# plt.grid()
# plt.show()


# y_true = y2_1

# # Compare Metrics
# mae_poly = mean_absolute_error(y_true, mdot_PR)
# mape_poly = mean_absolute_percentage_error(y_true, mdot_PR) * 100
# r2_poly = r2_score(y_true, mdot_PR)

# mae_lr = mean_absolute_error(y_true, y2_1_pred)
# mape_lr = mean_absolute_percentage_error(y_true, y2_1_pred) * 100
# r2_lr = r2_score(y_true, y2_1_pred)

# # Print Comparison Metrics
# print("Comparison of mdot_PR (Polynomial Regression) and y2_1_pred (Linear Regression):")
# print(f"Polynomial Regression - MAE: {mae_poly:.2f}, MAPE: {mape_poly:.2f}%, R²: {r2_poly:.2f}")
# print(f"Linear Regression      - MAE: {mae_lr:.2f}, MAPE: {mape_lr:.2f}%, R²: {r2_lr:.2f}")

# # Plot Ground Truth vs Predictions
# plt.figure(figsize=(12, 6))

# plt.plot(y_true, label="Ground Truth (y2_1_test)", color="black", linewidth=2)
# plt.plot(mdot_PR, label="mdot_PR (Polynomial Regression)", linestyle="--", color="blue")
# plt.plot(y2_1_pred, label="y2_1_pred (Linear Regression)", linestyle="--", color="red")

# plt.xlabel("Test Data Index")
# plt.ylabel("mdot [g/s]")
# plt.title("Comparison of Polynomial and Linear Regression Predictions")
# plt.legend()
# plt.grid()
# plt.show()


# y_true = y2_3

# # Compare Metrics
# mae_poly = mean_absolute_error(y_true, Pcooler1_PR)
# mape_poly = mean_absolute_percentage_error(y_true, Pcooler1_PR) * 100
# r2_poly = r2_score(y_true, Pcooler1_PR)

# mae_lr = mean_absolute_error(y_true, y2_3_pred)
# mape_lr = mean_absolute_percentage_error(y_true, y2_3_pred) * 100
# r2_lr = r2_score(y_true, y2_3_pred)

# # Print Comparison Metrics
# print("Comparison of Pcooler1_PR (Polynomial Regression) and y2_3_pred (Linear Regression):")
# print(f"Polynomial Regression - MAE: {mae_poly:.2f}, MAPE: {mape_poly:.2f}%, R²: {r2_poly:.2f}")
# print(f"Linear Regression      - MAE: {mae_lr:.2f}, MAPE: {mape_lr:.2f}%, R²: {r2_lr:.2f}")

# # Plot Ground Truth vs Predictions
# plt.figure(figsize=(12, 6))

# plt.plot(y_true, label="Ground Truth (y2_3_test)", color="black", linewidth=2)
# plt.plot(Pcooler2_PR, label="Pcooler1_PR (Polynomial Regression)", linestyle="--", color="blue")
# plt.plot(y2_3_pred, label="y2_3_pred (Linear Regression)", linestyle="--", color="red")

# plt.xlabel("Test Data Index")
# plt.ylabel("Pcooling1 [W]")
# plt.title("Comparison of Polynomial and Linear Regression Predictions")
# plt.legend()
# plt.grid()
# plt.show()
# Ground truth
# y_true = y1_3

# # Compare Metrics
# mae_poly = mean_absolute_error(y_true, Pcooler1_PR)
# mape_poly = mean_absolute_percentage_error(y_true, Pcooler1_PR) * 100
# r2_poly = r2_score(y_true, Pcooler1_PR)

# mae_lr = mean_absolute_error(y_true, y1_3_pred)
# mape_lr = mean_absolute_percentage_error(y_true, y1_3_pred) * 100
# r2_lr = r2_score(y_true, y1_3_pred)

# # Print Comparison Metrics
# print("Comparison of Pcooler1_PR (Polynomial Regression) and y1_3_pred (Linear Regression):")
# print(f"Polynomial Regression - MAE: {mae_poly:.2f}, MAPE: {mape_poly:.2f}%, R²: {r2_poly:.2f}")
# print(f"Linear Regression      - MAE: {mae_lr:.2f}, MAPE: {mape_lr:.2f}%, R²: {r2_lr:.2f}")

# # Plot Ground Truth vs Predictions
# plt.figure(figsize=(12, 6))

# plt.plot(y_true, label="Ground Truth (y1_3_test)", color="black", linewidth=2)
# plt.plot(Pcooler1_PR, label="Pcooler1_PR (Polynomial Regression)", linestyle="--", color="blue")
# plt.plot(y1_3_pred, label="y1_3_pred (Linear Regression)", linestyle="--", color="red")

# plt.xlabel("Test Data Index")
# plt.ylabel("Pcooling1 [W]")
# plt.title("Comparison of Polynomial and Linear Regression Predictions")
# plt.legend()
# plt.grid()
# plt.show()

