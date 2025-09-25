import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import csv
import numpy as np
data = {
    "Tin [K]": [],
    "pin [pa]": [],
    "pout [pa]": [],
    "Th_wall [K]": [],
    "Tw_in [K]": [],
    "omega [rpm]": [],
    "mdot [kg/s]": [],
    "Pcomb [W]": [],
    "Pheating [W]": [],
    "Pcooling [W]": [],
    "Pmotor [W]": [],
    "Tout [K]": []
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
        data['mdot [kg/s]'].append(float(row['mdot [g/s]'])*10**(-3))
        data['Pcomb [W]'].append(float(row['Pcomb [W]']))
        data['Pheating [W]'].append(float(row['Pheating [W]']))
        data['Pcooling [W]'].append(float(row['Pcooling [W]']))
        data['Pmotor [W]'].append(float(row['Pmotor [W]']))
        data['Tout [K]'].append(float(row['Tout [°C]']) + 273.15)
        # if counter > 52:
        #     break

Pr = [s/r for s,r in zip(data['pout [pa]'][:], data['pin [pa]'][:])]
Tr = [s/r for s,r in zip(data['Th_wall [K]'][:], data['Tw_in [K]'][:])]

# X_array = np.array([Pr, Tr,data['omega [rpm]'][:]])
X_array = np.array([data['pin [pa]'],data['pout [pa]'], data['omega [rpm]'][:], data['Th_wall [K]'], data['Tw_in [K]']])
X = X_array.T
y1 = np.array(data['mdot [kg/s]'])

# Split the dataset into training and test sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size = 0.2, random_state = 0)

# Feature scaling

# Building the ANN model
model1_ANN = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X1_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),  # Added layer
    tf.keras.layers.Dense(128, activation='relu'),  # Added layer
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer
])

model1_ANN.compile(optimizer='adam', loss='mean_squared_error')

model1_ANN.fit(X1_train, y1_train, batch_size=20, epochs= 100, validation_split=0.2)  # Adjust batch size here

y1_pred = model1_ANN.predict(X1_test)

print(y1_pred)

