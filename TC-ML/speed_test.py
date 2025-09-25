import csv
import tensorflow as tf
import numpy as np
from utils import get_state, CP, T0, p0, h0, s0
import matplotlib.pyplot as plt
import os
import joblib
from data_filling import(data, a_pcharged, a_Tr, a_Pr)

GP_models_dir = 'GP_models'
models_GP = [joblib.load(os.path.join(GP_models_dir, f'GP_model_{i+1}.pkl')) for i in range(6)]

NN_models_dir = 'NN_models'
models_NN = [tf.keras.models.load_model(os.path.join(NN_models_dir, f'NN_model_{i+1}.h5')) for i in range(6)]

scalers_dir = 'scalers'
scaler_X = joblib.load(os.path.join(scalers_dir, 'scaler_X.pkl'))
scalers_y = [joblib.load(os.path.join(scalers_dir, f'scaler_y_{i+1}.pkl')) for i in range(6)]

import time

Pr = 1.35
p1 = 40e5
p2 = p1 * Pr
pcharged = np.sqrt(p1 * p2)
Theater = 973.15
Twi = 293.15
Tr = Theater/Twi
omega = 150

# store elapsed times and predictions
times = []
all_preds = []

for k in range(100):
    t0 = time.perf_counter()

    # Construct input sample
    state = get_state(CP.PQ_INPUTS, p1, 1)
    T1 = state.T() + 3
    state = get_state(CP.PT_INPUTS, p1, T1)
    h1 = state.hmass()
    s1 = state.smass()
    test1 = np.array([[p1, p2, omega, Theater, Twi, T1]])  # shape (1, 6)

    test1_scaled = scaler_X.transform(test1)

    # # GP predictions for all 6 outputs
    # predictions_GP = [model.predict(test1_scaled) for model in models_GP]
    # y_pred_real_GP = [
    #     scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()[0]
    #     for pred, scaler in zip(predictions_GP, scalers_y)
    # ]

    predictions_NN = [model.predict(test1_scaled) for model in models_NN]
    y_pred_real_NN = [scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()[0]
                      for pred, scaler in zip(predictions_NN, scalers_y)]

    y_pred_real_NN[0]

    all_preds.append(y_pred_real_NN)
    times.append(time.perf_counter() - t0)  # record iteration time

    # predictions_NN = [model.predict(test1_scaled) for model in models_NN]
    # y_pred_real_NN = [scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()[0]
    #                   for pred, scaler in zip(predictions_NN, scalers_y)]

    # # # Assign predictions to each target array
    # a_mdot[k]   = y_pred_real_NN[0]
    # (Optional) print every 1000 iterations
    # if (k + 1) % 1000 == 0:
    #     print(f"Iteration {k+1}: time {times[-1]*1e3:.3f} ms, first target = {y_pred_real_NN[0]:.3f}")

# After loop: summary
import numpy as np
times = np.array(times)
print(f"\nAverage inference time per iteration: {times.mean()*1e3:.3f} ms")
print(f"Std dev: {times.std()*1e3:.3f} ms")
print(f"p95: {np.percentile(times,95)*1e3:.3f} ms")
