# -------------------------
# IMPORTS
# -------------------------
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from data_filling import (X_scaled, y_scaled)
# -------------------------
# SPLIT DATA
# -------------------------
X_train, X_test, idx_train, idx_test = train_test_split(X_scaled, np.arange(len(X_scaled)), test_size=0.2)
y_train = [y[idx_train] for y in y_scaled]
y_test = [y[idx_test] for y in y_scaled]

n_features = X_scaled.shape[1]
# ------------------------------------
# Train NN
# ------------------------------------
models_NN= []
for yt in y_train:
    # NN
    NN = tf.keras.Sequential([
        tf.keras.layers.Dense(300, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    NN.compile(optimizer='adam', loss='mean_squared_error')

    print(f"\nTraining NN model for output {len(models_NN)+1}")
    NN.fit(
        X_train, yt,
        batch_size=20,
        epochs=100,
        validation_split=0.2,
        verbose=0)
    
    models_NN.append(NN)  # ✅ This was missing
    
# ------------------------------------
# Create a directory to save models
# ------------------------------------
save_dir = 'NN_models'
os.makedirs(save_dir, exist_ok=True)

# Save NN models
for i, NN in enumerate(models_NN):
    NN.save(os.path.join(save_dir, f'NN_model_{i+1}.h5'))