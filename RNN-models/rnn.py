import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_filling import train_data, test_data
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Set random seeds for reproducibility
torch.manual_seed(42)  # Seed for PyTorch (affects model initialization)
np.random.seed(42)     # Seed for NumPy (affects any NumPy random operations)

# ------------------------
# Prepare Data
# ------------------------
X_train = np.column_stack([train_data["pc"], train_data["Tw_out"]])  # States
U_train = np.column_stack([train_data["Hpev"], train_data["omegab"],
                           train_data["mw_dot"], train_data["Tw_in"]])  # Inputs
y_train = np.column_stack([train_data["pc"], train_data["Tw_out"]])  # Target (next states)

X_test = np.column_stack([test_data["pc"], test_data["Tw_out"]])
U_test = np.column_stack([test_data["Hpev"], test_data["omegab"],
                          test_data["mw_dot"], test_data["Tw_in"]])
y_test = np.column_stack([test_data["pc"], test_data["Tw_out"]])

# Concatenate state and input features
input_train = np.hstack([X_train, U_train])
input_test = np.hstack([X_test, U_test])

# Normalize the input and target data
scaler_input = StandardScaler()
scaler_output = StandardScaler()

input_train = scaler_input.fit_transform(input_train)
y_train = scaler_output.fit_transform(y_train)
input_test = scaler_input.transform(input_test)
y_test = scaler_output.transform(y_test)

# Split training data into train and validation sets (no shuffling already)
train_idx, val_idx = train_test_split(range(len(input_train)), test_size=0.2, shuffle=False)
input_train_split = input_train[train_idx]
y_train_split = y_train[train_idx]
input_val = input_train[val_idx]
y_val = y_train[val_idx]

# Function to create sequences for multi-step prediction
def create_sequences(input_data, output_data, seq_length, pred_length):
    X, y = [], []
    for i in range(len(input_data) - seq_length - pred_length + 1):
        X.append(input_data[i : i + seq_length])  # Input sequence
        y.append(output_data[i + seq_length : i + seq_length + pred_length])  # Multi-step target
    return np.array(X), np.array(y)

# Parameters
sequence_length = 10  # Number of time steps per sequence
prediction_length = 1  # Number of steps to predict ahead
batch_size = 32
use_batch = False  # Toggle batch processing on (True) or off (False)

# Prepare training, validation, and test sequences
input_train_seq, y_train_seq = create_sequences(input_train_split, y_train_split, sequence_length, prediction_length)
input_val_seq, y_val_seq = create_sequences(input_val, y_val, sequence_length, prediction_length)
input_test_seq, y_test_seq = create_sequences(input_test, y_test, sequence_length, prediction_length)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(input_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
X_val_tensor = torch.tensor(input_val_seq, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
X_test_tensor = torch.tensor(input_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

# Prepare Datasets (no shuffling here)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Use shuffle=True **only** for training
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------
# Define Elman RNN Model
# ------------------------
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_length, num_layers, dropout):
        super(RNNModel, self).__init__()
        self.pred_length = pred_length
        self.output_size = output_size
        self.lstm = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # PyTorch requires dropout only if num_layers > 1
        )
        self.fc = nn.Linear(hidden_size, output_size * pred_length)

    def forward(self, x):
        out, _ = self.lstm(x)                # out: (batch, seq_len, hidden)
        out = self.fc(out[:, -1, :])         # use last time step
        out = out.view(-1, self.pred_length, self.output_size)
        return out


# Hyperparameters
input_size = 6    # Number of input features (state + inputs)
output_size = 2   # Number of output features per time step (pc, Tw_out)
hidden_size = 64
num_layers = 1
num_epochs = 1000
learning_rate = 0.001
dropout = 0.1

# Initialize RNN model, loss, optimizer, scheduler
model = RNNModel(input_size, hidden_size, output_size, prediction_length, num_layers, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# # ------------------------
# # Training Loop with Early Stopping
# # ------------------------
# best_val_loss = float('inf')
# patience = 40
# # Number of epochs to wait for improvement
# patience_counter = 0
# train_losses = []
# val_losses = []

# for epoch in range(num_epochs):
#     model.train()
#     total_train_loss = 0.0
#     if use_batch:  # Batch processing
#         for X_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item() * X_batch.size(0)
#         train_loss = total_train_loss / len(train_dataset)
#     else:  # Non-batch processing
#         optimizer.zero_grad()
#         outputs = model(X_train_tensor)
#         loss = criterion(outputs, y_train_tensor)
#         loss.backward()
#         optimizer.step()
#         train_loss = loss.item()

#     # Validation loss
#     model.eval()
#     with torch.no_grad():
#         if use_batch:  # Batch processing for validation
#             total_val_loss = 0.0
#             for X_val_batch, y_val_batch in val_loader:
#                 val_outputs = model(X_val_batch)
#                 val_loss = criterion(val_outputs, y_val_batch)
#                 total_val_loss += val_loss.item() * X_val_batch.size(0)
#             val_loss = total_val_loss / len(val_dataset)
#         else:  # Non-batch processing for validation
#             val_outputs = model(X_val_tensor)
#             val_loss = criterion(val_outputs, y_val_tensor).item()

#     # Store losses for plotting
#     train_losses.append(train_loss)
#     val_losses.append(val_loss)

#     # Early stopping
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         patience_counter = 0
#         torch.save(model.state_dict(), "best_rnn_model.pth")  # Save best model
#     else:
#         patience_counter += 1
#         if patience_counter >= patience:
#             print(f"Early stopping triggered after epoch {epoch + 1}")
#             break

#     scheduler.step(val_loss)  # Adjust learning rate based on validation loss

#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# # Load the best model
# model.load_state_dict(torch.load("best_rnn_model.pth"))
# # ------------------------
# # Professional Plot: Training & Validation Loss
# # ------------------------
# import matplotlib.ticker as ticker

# plt.figure(figsize=(8, 5))
# plt.plot(train_losses, label="Training Loss", linewidth=2.5, color="steelblue")
# plt.plot(val_losses, label="Validation Loss", linewidth=2.5, color="darkorange", linestyle="--")

# plt.xlabel("Epoch", fontsize=14)
# plt.ylabel("MSE Loss", fontsize=14)
# # plt.title("Training and Validation Loss", fontsize=16)
# plt.legend(loc="upper right", fontsize=12, frameon=False)

# # Improve tick formatting
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # integer ticks for epochs
# plt.grid(True, which='major', linestyle=':', linewidth=1, alpha=0.7)

# # Tight layout for articles
# plt.tight_layout()

# # Save if needed
# # plt.savefig("training_loss_plot.pdf", bbox_inches="tight")  # vector format
# # plt.savefig("training_loss_plot.png", dpi=300)

# plt.show()
# # ------------------------
# # Evaluation Loop
# # ------------------------
# model.eval()
# with torch.no_grad():
#     train_predictions = model(X_train_tensor)
#     train_loss = criterion(train_predictions, y_train_tensor).item()

#     val_predictions = model(X_val_tensor)
#     val_loss = criterion(val_predictions, y_val_tensor).item()

#     test_predictions = model(X_test_tensor)
#     test_loss = criterion(test_predictions, y_test_tensor).item()

# print(f"Training MSE: {train_loss:.4f}")
# print(f"Validation MSE: {val_loss:.4f}")
# print(f"Test MSE: {test_loss:.4f}")

# # ------------------------
# # Save the Model and Scalers
# # ------------------------
# torch.save(model.state_dict(), "RNN2/rnn_model.pth")
# joblib.dump(scaler_input, "RNN2/scaler_input.pkl")
# joblib.dump(scaler_output, "RNN2/scaler_output.pkl")

# print("Model and scalers saved.")

# # ------------------------
# # Prepare Test Data
# # ------------------------
# X_test = np.column_stack([test_data["pc"], test_data["Tw_out"]])
# U_test = np.column_stack([test_data["Hpev"], test_data["omegab"],
#                           test_data["mw_dot"], test_data["Tw_in"]])
# y_test = np.column_stack([test_data["pc"], test_data["Tw_out"]])

# input_test = np.hstack([X_test, U_test])
# input_test = scaler_input.transform(input_test)
# y_test_scaled = scaler_output.transform(y_test)

# # Create test sequences
# def create_sequences(input_data, output_data, seq_length, pred_length):
#     X, y = [], []
#     for i in range(len(input_data) - seq_length - pred_length + 1):
#         X.append(input_data[i : i + seq_length])
#         y.append(output_data[i + seq_length : i + seq_length + pred_length])
#     return np.array(X), np.array(y)

# input_test_seq, y_test_seq = create_sequences(input_test, y_test_scaled, sequence_length, prediction_length)

# # Convert to tensors
# X_test_tensor = torch.tensor(input_test_seq, dtype=torch.float32)

# # ------------------------
# # Predict
# # ------------------------
# with torch.no_grad():
#     test_predictions = model(X_test_tensor)

# # Denormalize only the last predicted step
# test_predictions_last = test_predictions[:, -1, :].numpy()
# y_test_true_last = y_test_seq[:, -1, :]

# test_predictions_denorm = scaler_output.inverse_transform(test_predictions_last)
# y_test_denorm = scaler_output.inverse_transform(y_test_true_last)

# # ------------------------
# # Evaluate
# # ------------------------
# def calculate_metrics(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     mape = mean_absolute_percentage_error(y_true, y_pred) * 100
#     r2 = r2_score(y_true, y_pred)
#     return mse, mape, r2


# mse_pc, mape_pc, r2_pc = calculate_metrics(y_test_denorm[:, 0], test_predictions_denorm[:, 0])
# mse_Tw_out, mape_Tw_out, r2_Tw_out = calculate_metrics(y_test_denorm[:, 1], test_predictions_denorm[:, 1])

# # ------------------------
# # Plot
# # ------------------------
# fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# N_half = 2681  # Index separating physical model data and real data

# # -------- High-pressure --------
# ax[0].plot(y_test_denorm[:N_half, 0], label="Real data", color="blue", linewidth=2)
# ax[0].plot(np.arange(N_half, len(y_test_denorm)), y_test_denorm[N_half:, 0], label="Physical model data", color="green", linewidth=2)
# ax[0].plot(test_predictions_denorm[:, 0],
#            label=f"RNN Prediction (MSE: {mse_pc:.2f} Bar², R²: {r2_pc:.3f})",
#            linestyle="dashed", color="orange", linewidth=2)

# # Vertical separator
# ax[0].axvline(x=N_half, color='black', linestyle='--', linewidth=1)

# ax[0].set_ylabel("$p_c$ [Bar]", fontsize=14)
# ax[0].legend(loc="upper left", fontsize=12)
# ax[0].tick_params(axis="both", labelsize=12)
# ax[0].grid(True)

# # -------- Water Outlet Temperature --------
# y_true_C = y_test_denorm[:, 1] - 273.15
# y_pred_C = test_predictions_denorm[:, 1] - 273.15

# ax[1].plot(np.arange(N_half), y_true_C[:N_half], label="Real data", color="blue", linewidth=2)
# ax[1].plot(np.arange(N_half, len(y_true_C)), y_true_C[N_half:], label="Physical model data", color="green", linewidth=2)
# ax[1].plot(y_pred_C,
#            label=f"RNN Prediction (MSE: {mse_Tw_out:.2f} K², R²: {r2_Tw_out:.3f})",
#            linestyle="dashed", color="orange", linewidth=2)

# # Vertical separator
# ax[1].axvline(x=N_half, color='black', linestyle='--', linewidth=1)

# ax[1].set_xlabel("Time [s]", fontsize=14)
# ax[1].set_ylabel("$T_{w, out}$ [°C]", fontsize=14)
# ax[1].legend(loc="upper left", fontsize=12)
# ax[1].tick_params(axis="both", labelsize=12)
# ax[1].grid(True)

# plt.tight_layout()
# plt.show()


model = RNNModel(input_size, hidden_size, output_size, prediction_length, num_layers, dropout)
model.load_state_dict(torch.load("RNN2/rnn_model.pth"))
model.eval()

scaler_input = joblib.load("RNN2/scaler_input.pkl")
scaler_output = joblib.load("RNN2/scaler_output.pkl")
# Recursion

# Parameters
prediction_length = 500  # Steps to predict per cycle
total_steps = 10 * (sequence_length + prediction_length)  # Total simulation length
cycles = total_steps // (sequence_length + prediction_length)  # Number of cycles (e.g., 5 for 200 steps)

# Prepare the full input and normalize
X_test = np.column_stack([test_data["pc"], test_data["Tw_out"]])
U_test = np.column_stack([test_data["Hpev"], test_data["omegab"], test_data["mw_dot"], test_data["Tw_in"]])
y_test_true = np.column_stack([test_data["pc"], test_data["Tw_out"]])
input_test_full = np.hstack([X_test, U_test])
input_test_scaled = scaler_input.transform(input_test_full)
y_test_scaled = scaler_output.transform(y_test_true)

# Validate start_idx
start_idx = 0 # Starting index
max_idx = len(input_test_scaled) - sequence_length - prediction_length
if start_idx < 0 or start_idx > max_idx:
    raise ValueError(f"start_idx must be between 0 and {max_idx}, got {start_idx}")

# Initialize lists to store predictions and true values
all_recursive_preds = []
all_true_outputs = []
all_time_steps = []
all_input_time_steps = []
all_prediction_time_steps = []

model.eval()
with torch.no_grad():
    for cycle in range(cycles):
        # Calculate indices for current cycle
        cycle_start = start_idx + cycle * (sequence_length + prediction_length)
        input_end = cycle_start + sequence_length
        pred_end = input_end + prediction_length

        # Get initial sequence (known data)
        recursive_input = input_test_scaled[cycle_start:input_end].copy()  # shape: (20, 6)
        recursive_preds = []

        # Predict for prediction_length steps
        for step in range(prediction_length):
            input_tensor = torch.tensor(recursive_input[np.newaxis, :, :], dtype=torch.float32)  # (1, 20, 6)
            pred_scaled = model(input_tensor)[0, -1, :].numpy()  # shape: (2,)
            pred_real = scaler_output.inverse_transform(pred_scaled.reshape(1, -1))[0]
            recursive_preds.append(pred_real)
            control_input = input_test_scaled[input_end + step, 2:]  # shape: (4,)
            new_input = np.hstack([pred_scaled, control_input])  # shape: (6,)
            recursive_input = np.vstack([recursive_input[1:], new_input])

        # Store predictions and true outputs
        all_recursive_preds.append(recursive_preds)
        all_true_outputs.append(y_test_true[input_end:pred_end])
        all_time_steps.extend(range(cycle_start - start_idx, pred_end - start_idx))
        all_input_time_steps.extend(range(cycle_start - start_idx, input_end - start_idx))
        all_prediction_time_steps.extend(range(input_end - start_idx, pred_end - start_idx))

# Convert to arrays
all_recursive_preds = np.vstack(all_recursive_preds)  # shape: (100, 2) for 5 cycles
all_true_outputs = np.vstack(all_true_outputs)  # shape: (100, 2)

# Calculate MSE and R²
mse_pc = mean_squared_error(all_true_outputs[:, 0], all_recursive_preds[:, 0])
r2_pc = r2_score(all_true_outputs[:, 0], all_recursive_preds[:, 0])
mse_Tw_out = mean_squared_error(all_true_outputs[:, 1], all_recursive_preds[:, 1])
r2_Tw_out = r2_score(all_true_outputs[:, 1], all_recursive_preds[:, 1])

# Print results
print(f"Alternating Recursive pc - MSE: {mse_pc:.2f} Bar², R²: {r2_pc:.3f}")
print(f"Alternating Recursive Tw_out - MSE: {mse_Tw_out:.2f} K², R²: {r2_Tw_out:.3f}")

# Extract omegab and Hpev for plotting
omegab = U_test[start_idx:start_idx + total_steps, 1]
Hpev = U_test[start_idx:start_idx + total_steps, 0]

# Plot
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Combine input and predicted values for pc and Tw_out
model_pc = []
model_Tw_out = []
for cycle in range(cycles):
    cycle_start = start_idx + cycle * (sequence_length + prediction_length)
    input_end = cycle_start + sequence_length
    # Append input values (from y_test_true)
    model_pc.extend(y_test_true[cycle_start:input_end, 0])
    model_Tw_out.extend(y_test_true[cycle_start:input_end, 1] - 273.15)  # Convert to Celsius
    # Append predicted values
    model_pc.extend(all_recursive_preds[cycle * prediction_length:(cycle + 1) * prediction_length, 0])
    model_Tw_out.extend(all_recursive_preds[cycle * prediction_length:(cycle + 1) * prediction_length, 1] - 273.15)  # Convert to Celsius

model_pc = np.array(model_pc)
model_Tw_out = np.array(model_Tw_out)

# -------- High-pressure: $p_c$ --------
for cycle in range(cycles):
    cycle_start = cycle * (sequence_length + prediction_length)
    input_end = cycle_start + sequence_length
    pred_end = cycle_start + sequence_length + prediction_length

    # Input segment (green) — emphasized
    ax[0].plot(
        all_time_steps[cycle_start:input_end],
        model_pc[cycle_start:input_end],
        color="green", linewidth=3, linestyle="-", marker='o', markersize=5,
        label="Model Input $p_c$" if cycle == 0 else "", zorder=3
    )

    # Predicted segment (orange)
    ax[0].plot(
        all_time_steps[input_end:pred_end],
        model_pc[input_end:pred_end],
        color="orange", linewidth=2, linestyle="-",
        label=f"Model Predicted $p_c$ (MSE: {mse_pc:.2f} Bar², R²: {r2_pc:.3f})" if cycle == 0 else "", zorder=2
    )

# True data (blue dashed)
# Separate true data: experimental (first 1000) vs. physical model
exp_end = 2680
all_true_pc = y_test_true[start_idx:start_idx + total_steps, 0]

# Plot experimental
ax[0].plot(
    all_time_steps[:exp_end],
    all_true_pc[:exp_end],
    label="Experimental $p_c$", color="blue", linestyle="--", linewidth=2.5, marker='x', markersize=4, zorder=1
)

# Plot physical model
ax[0].plot(
    all_time_steps[exp_end:],
    all_true_pc[exp_end:],
    label="Physical Model $p_c$", color="purple", linestyle="--", linewidth=2.5, marker='x', markersize=4, zorder=1
)


# Vertical boundaries
for cycle in range(cycles):
    boundary = (cycle + 1) * sequence_length + cycle * prediction_length
    ax[0].axvline(x=boundary, color='red', linewidth=2, linestyle=':', label="Input/Prediction Boundary" if cycle == 0 else "")

ax[0].set_ylabel("$p_c$ [Bar]", fontsize=14)
ax[0].legend(loc="upper left", fontsize=12)
ax[0].grid(True)
ax[0].tick_params(axis="both", labelsize=12)

# -------- Water Outlet Temperature: $T_{w,out}$ --------
true_output_C = y_test_true[start_idx:start_idx + total_steps, 1] - 273.15

for cycle in range(cycles):
    cycle_start = cycle * (sequence_length + prediction_length)
    input_end = cycle_start + sequence_length
    pred_end = cycle_start + sequence_length + prediction_length

    # Input segment (green) — emphasized
    ax[1].plot(
        all_time_steps[cycle_start:input_end],
        model_Tw_out[cycle_start:input_end],
        color="green", linewidth=3, linestyle="-", marker='o', markersize=5,
        label="Model Input $T_{w,out}$" if cycle == 0 else "", zorder=3
    )

    # Predicted segment (orange)
    ax[1].plot(
        all_time_steps[input_end:pred_end],
        model_Tw_out[input_end:pred_end],
        color="orange", linewidth=2, linestyle="-",
        label=f"Model Predicted $T_{{w,out}}$ (MSE: {mse_Tw_out:.2f} K², R²: {r2_Tw_out:.3f})" if cycle == 0 else "", zorder=2
    )

# True data (blue dashed)
# Separate true data
true_output_C = y_test_true[start_idx:start_idx + total_steps, 1] - 273.15

# Plot experimental
ax[1].plot(
    all_time_steps[:exp_end],
    true_output_C[:exp_end],
    label="Experimental $T_{w,out}$", color="blue", linestyle="--", linewidth=2.5, marker='x', markersize=4, zorder=1
)

# Plot physical model
ax[1].plot(
    all_time_steps[exp_end:],
    true_output_C[exp_end:],
    label="Physical Model $T_{w,out}$", color="purple", linestyle="--", linewidth=2.5, marker='x', markersize=4, zorder=1
)

for cycle in range(cycles):
    boundary = (cycle + 1) * sequence_length + cycle * prediction_length
    ax[1].axvline(x=boundary, color='red', linewidth=2, linestyle=':', label="Input/Prediction Boundary" if cycle == 0 else "")

ax[1].set_ylabel("$T_{w,out}$ [°C]", fontsize=14)
ax[1].set_xlabel("Time [s]", fontsize=14)
ax[1].legend(loc="upper left", fontsize=12)
ax[1].grid(True)
ax[1].tick_params(axis="both", labelsize=12)

# Optimize spacing
plt.tight_layout()  # Automatically adjust subplots to fit figure area

# Optional: fine-tune spacing between subplots and borders
plt.subplots_adjust(top=0.96, bottom=0.08, left=0.08, right=0.98, hspace=0.25)

# Show final plot
plt.show()
