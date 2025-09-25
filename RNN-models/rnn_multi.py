import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.ticker as ticker
import os
from data_filling import train_data, test_data

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
# Normalize states and controls separately
scaler_state = StandardScaler()
scaler_control = StandardScaler()
scaler_output = StandardScaler()

X_train_scaled = scaler_state.fit_transform(X_train)
U_train_scaled = scaler_control.fit_transform(U_train)
y_train_scaled = scaler_output.fit_transform(y_train)
X_test_scaled = scaler_state.transform(X_test)
U_test_scaled = scaler_control.transform(U_test)
y_test_scaled = scaler_output.transform(y_test)

# Split training data into train and validation sets
train_idx, val_idx = train_test_split(range(len(X_train_scaled)), test_size=0.2, shuffle=False)
X_train_split = X_train_scaled[train_idx]
U_train_split = U_train_scaled[train_idx]
y_train_split = y_train_scaled[train_idx]
X_val = X_train_scaled[val_idx]
U_val = U_train_scaled[val_idx]
y_val = y_train_scaled[val_idx]

# Function to create NARX sequences
def create_narx_sequences(X, U, y, seq_length, pred_length):
    X_seq, U_past_seq, U_future_seq, y_seq = [], [], [], []
    for i in range(len(X) - seq_length - pred_length + 1):
        X_seq.append(X[i:i + seq_length])  # Past states
        U_past_seq.append(U[i:i + seq_length])  # Past controls
        U_future_seq.append(U[i + seq_length:i + seq_length + pred_length])  # Future controls
        y_seq.append(y[i + seq_length:i + seq_length + pred_length])  # Targets
    return np.array(X_seq), np.array(U_past_seq), np.array(U_future_seq), np.array(y_seq)

# Parameters
sequence_length = 10
prediction_length = 20
batch_size = 32
use_batch = False

# Prepare sequences
input_train_seq, U_past_train_seq, U_future_train_seq, y_train_seq = create_narx_sequences(
    X_train_split, U_train_split, y_train_split, sequence_length, prediction_length)
input_val_seq, U_past_val_seq, U_future_val_seq, y_val_seq = create_narx_sequences(
    X_val, U_val, y_val, sequence_length, prediction_length)
input_test_seq, U_past_test_seq, U_future_test_seq, y_test_seq = create_narx_sequences(
    X_test_scaled, U_test_scaled, y_test_scaled, sequence_length, prediction_length)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(input_train_seq, dtype=torch.float32)
U_past_train_tensor = torch.tensor(U_past_train_seq, dtype=torch.float32)
U_future_train_tensor = torch.tensor(U_future_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
X_val_tensor = torch.tensor(input_val_seq, dtype=torch.float32)
U_past_val_tensor = torch.tensor(U_past_val_seq, dtype=torch.float32)
U_future_val_tensor = torch.tensor(U_future_val_seq, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
X_test_tensor = torch.tensor(input_test_seq, dtype=torch.float32)
U_past_test_tensor = torch.tensor(U_past_test_seq, dtype=torch.float32)
U_future_test_tensor = torch.tensor(U_future_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

# Prepare Datasets
train_dataset = TensorDataset(X_train_tensor, U_past_train_tensor, U_future_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, U_past_val_tensor, U_future_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, U_past_test_tensor, U_future_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------
# Define NARX Pipeline Model
# ------------------------
class NARXPipeline(nn.Module):
    def __init__(self, state_input_size, control_input_size, hidden_size, output_size, pred_length, num_layers, dropout):
        super(NARXPipeline, self).__init__()
        self.pred_length = pred_length
        self.output_size = output_size
        self.past_rnn = nn.RNN(
            input_size=state_input_size + control_input_size,  # 2 + 4
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.future_rnn = nn.RNN(
            input_size=hidden_size + control_input_size,  # hidden_size + 4
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size * pred_length)

    def forward(self, X, U_past, U_future):
        past_input = torch.cat((X, U_past), dim=2)  # (batch, seq_length, 6)
        past_out, _ = self.past_rnn(past_input)  # (batch, seq_length, hidden_size)
        past_out = past_out[:, -1, :]  # (batch, hidden_size)
        past_out = past_out.unsqueeze(1).repeat(1, self.pred_length, 1)  # (batch, pred_length, hidden_size)
        future_input = torch.cat((past_out, U_future), dim=2)  # (batch, pred_length, hidden_size + 4)
        future_out, _ = self.future_rnn(future_input)  # (batch, pred_length, hidden_size)
        out = self.fc(future_out[:, -1, :])  # (batch, output_size * pred_length)
        out = out.view(-1, self.pred_length, self.output_size)  # (batch, pred_length, output_size)
        return out

# Hyperparameters
state_input_size = 2  # pc, Tw_out
control_input_size = 4  # Hpev, omegab, mw_dot, Tw_in
output_size = 2
hidden_size = 64
num_layers = 1
num_epochs = 200
learning_rate = 0.001
dropout = 0.1

# Initialize model, loss, optimizer, scheduler
model = NARXPipeline(state_input_size, control_input_size, hidden_size, output_size, prediction_length, num_layers, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# ------------------------
# Training Loop with Early Stopping
# ------------------------
best_val_loss = float('inf')
patience = 40
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    if use_batch:  # Batch processing
        for X_batch, U_past_batch, U_future_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch, U_past_batch, U_future_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)
        train_loss = total_train_loss / len(train_dataset)
    else:  # Non-batch processing
        optimizer.zero_grad()
        outputs = model(X_train_tensor, U_past_train_tensor, U_future_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

    # Validation loss
    model.eval()
    with torch.no_grad():
        if use_batch:  # Batch processing for validation
            total_val_loss = 0.0
            for X_val_batch, U_past_val_batch, U_future_val_batch, y_val_batch in val_loader:
                val_outputs = model(X_val_batch, U_past_val_batch, U_future_val_batch)
                val_loss = criterion(val_outputs, y_val_batch)
                total_val_loss += val_loss.item() * X_val_batch.size(0)
            val_loss = total_val_loss / len(val_dataset)
        else:  # Non-batch processing for validation
            val_outputs = model(X_val_tensor, U_past_val_tensor, U_future_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

    # Store losses for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        os.makedirs("NARX_pipeline", exist_ok=True)  # Create directory if it doesn't exist
        torch.save(model.state_dict(), "NARX_pipeline/best_narx_pipeline.pth")  # Save best model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    scheduler.step(val_loss)  # Adjust learning rate based on validation loss

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Load the best model
model.load_state_dict(torch.load("NARX_pipeline/best_narx_pipeline.pth"))

# ------------------------
# Plot Training & Validation Loss
# ------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss", linewidth=2.5, color="steelblue")
plt.plot(val_losses, label="Validation Loss", linewidth=2.5, color="darkorange", linestyle="--")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("MSE Loss", fontsize=14)
plt.legend(loc="upper right", fontsize=12, frameon=False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.grid(True, which='major', linestyle=':', linewidth=1, alpha=0.7)
plt.tight_layout()
plt.show()

# ------------------------
# Evaluation Loop
# ------------------------
model.eval()
with torch.no_grad():
    train_predictions = model(X_train_tensor, U_past_train_tensor, U_future_train_tensor)
    train_loss = criterion(train_predictions, y_train_tensor).item()

    val_predictions = model(X_val_tensor, U_past_val_tensor, U_future_val_tensor)
    val_loss = criterion(val_predictions, y_val_tensor).item()

    test_predictions = model(X_test_tensor, U_past_test_tensor, U_future_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor).item()

print(f"Training MSE: {train_loss:.4f}")
print(f"Validation MSE: {val_loss:.4f}")
print(f"Test MSE: {test_loss:.4f}")

# ------------------------
# Save the Model and Scalers
# ------------------------
os.makedirs("NARX_pipeline", exist_ok=True)  # Create directory if it doesn't exist
torch.save(model.state_dict(), "NARX_pipeline/narx_pipeline.pth")
joblib.dump(scaler_state, "NARX_pipeline/scaler_state.pkl")
joblib.dump(scaler_control, "NARX_pipeline/scaler_control.pkl")
joblib.dump(scaler_output, "NARX_pipeline/scaler_output.pkl")
print("Model and scalers saved.")

# ------------------------
# Prepare Test Data
# ------------------------
X_test = np.column_stack([test_data["pc"], test_data["Tw_out"]])
U_test = np.column_stack([test_data["Hpev"], test_data["omegab"], test_data["mw_dot"], test_data["Tw_in"]])
y_test = np.column_stack([test_data["pc"], test_data["Tw_out"]])

X_test_scaled = scaler_state.transform(X_test)
U_test_scaled = scaler_control.transform(U_test)
y_test_scaled = scaler_output.transform(y_test)

# Create test sequences
def create_narx_sequences(X, U, y, seq_length, pred_length):
    X_seq, U_past_seq, U_future_seq, y_seq = [], [], [], []
    for i in range(len(X) - seq_length - pred_length + 1):
        X_seq.append(X[i:i + seq_length])
        U_past_seq.append(U[i:i + seq_length])
        U_future_seq.append(U[i + seq_length:i + seq_length + pred_length])
        y_seq.append(y[i + seq_length:i + seq_length + pred_length])
    return np.array(X_seq), np.array(U_past_seq), np.array(U_future_seq), np.array(y_seq)

input_test_seq, U_past_test_seq, U_future_test_seq, y_test_seq = create_narx_sequences(X_test_scaled, U_test_scaled, y_test_scaled, sequence_length, prediction_length)

# Convert to tensors
X_test_tensor = torch.tensor(input_test_seq, dtype=torch.float32)
U_past_test_tensor = torch.tensor(U_past_test_seq, dtype=torch.float32)
U_future_test_tensor = torch.tensor(U_future_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

# ------------------------
# Predict
# ------------------------
with torch.no_grad():
    test_predictions = model(X_test_tensor, U_past_test_tensor, U_future_test_tensor)

# Denormalize only the last predicted step
test_predictions_last = test_predictions[:, -1, :].numpy()
y_test_true_last = y_test_seq[:, -1, :]

test_predictions_denorm = scaler_output.inverse_transform(test_predictions_last)
y_test_denorm = scaler_output.inverse_transform(y_test_true_last)

# ------------------------
# Evaluate
# ------------------------
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, mape, r2

mse_pc, mape_pc, r2_pc = calculate_metrics(y_test_denorm[:, 0], test_predictions_denorm[:, 0])
mse_Tw_out, mape_Tw_out, r2_Tw_out = calculate_metrics(y_test_denorm[:, 1], test_predictions_denorm[:, 1])

# ------------------------
# Plot
# ------------------------
fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

N_half = 2681  # Index separating physical model data and real data

# -------- High-pressure --------
ax[0].plot(y_test_denorm[:N_half, 0], label="Real data", color="blue", linewidth=2)
ax[0].plot(np.arange(N_half, len(y_test_denorm)), y_test_denorm[N_half:, 0], label="Physical model data", color="green", linewidth=2)
ax[0].plot(test_predictions_denorm[:, 0],
           label=f"NARX Prediction (MSE: {mse_pc:.2f} Bar², R²: {r2_pc:.3f})",
           linestyle="dashed", color="orange", linewidth=2)

# Vertical separator
ax[0].axvline(x=N_half, color='black', linestyle='--', linewidth=1)

ax[0].set_ylabel("$p_c$ [Bar]", fontsize=14)
ax[0].legend(loc="upper left", fontsize=12)
ax[0].tick_params(axis="both", labelsize=12)
ax[0].grid(True)

# -------- Water Outlet Temperature --------
y_true_C = y_test_denorm[:, 1] - 273.15
y_pred_C = test_predictions_denorm[:, 1] - 273.15

ax[1].plot(np.arange(N_half), y_true_C[:N_half], label="Real data", color="blue", linewidth=2)
ax[1].plot(np.arange(N_half, len(y_true_C)), y_true_C[N_half:], label="Physical model data", color="green", linewidth=2)
ax[1].plot(y_pred_C,
           label=f"NARX Prediction (MSE: {mse_Tw_out:.2f} K², R²: {r2_Tw_out:.3f})",
           linestyle="dashed", color="orange", linewidth=2)

# Vertical separator
ax[1].axvline(x=N_half, color='black', linestyle='--', linewidth=1)

ax[1].set_xlabel("Time [s]", fontsize=14)
ax[1].set_ylabel("$T_{w,out}$ [°C]", fontsize=14)
ax[1].legend(loc="upper left", fontsize=12)
ax[1].tick_params(axis="both", labelsize=12)
ax[1].grid(True)

plt.tight_layout()
plt.show()

# Load the model and scalers
model = NARXPipeline(state_input_size, control_input_size, hidden_size, output_size, prediction_length, num_layers, dropout)
model.load_state_dict(torch.load("NARX_pipeline/narx_pipeline.pth"))
model.eval()
scaler_state = joblib.load("NARX_pipeline/scaler_state.pkl")
scaler_control = joblib.load("NARX_pipeline/scaler_control.pkl")
scaler_output = joblib.load("NARX_pipeline/scaler_output.pkl")
print("Model and scalers loaded.")
# ------------------------
# Recursive Prediction
# ------------------------
total_prediction_length = 500  # Validate for 500 steps
total_steps = 10 * (sequence_length + total_prediction_length)
cycles = total_steps // (sequence_length + total_prediction_length)
start_idx = 0
max_idx = len(X_test) - sequence_length - total_prediction_length
if start_idx < 0 or start_idx > max_idx:
    raise ValueError(f"start_idx must be between 0 and {max_idx}, got {start_idx}")

all_recursive_preds = []
all_true_outputs = []
all_time_steps = []
all_input_time_steps = []
all_prediction_time_steps = []

model.eval()
with torch.no_grad():
    for cycle in range(cycles):
        cycle_start = start_idx + cycle * (sequence_length + total_prediction_length)
        input_end = cycle_start + sequence_length
        pred_end = input_end + total_prediction_length

        recursive_states = X_test[cycle_start:input_end].copy()
        recursive_past_controls = U_test[cycle_start:input_end].copy()
        recursive_preds = []

        for step in range(0, total_prediction_length, prediction_length):
            future_start = input_end + step
            future_end = future_start + prediction_length
            if future_end > len(U_test):
                break

            U_future = U_test[future_start:future_end]
            if U_future.shape[0] < prediction_length:
                U_future = np.pad(U_future, ((0, prediction_length - U_future.shape[0]), (0, 0)), mode='edge')

            X_scaled = scaler_state.transform(recursive_states)
            U_past_scaled = scaler_control.transform(recursive_past_controls)
            U_future_scaled = scaler_control.transform(U_future)

            X_tensor = torch.tensor(X_scaled[np.newaxis, :, :], dtype=torch.float32)
            U_past_tensor = torch.tensor(U_past_scaled[np.newaxis, :, :], dtype=torch.float32)
            U_future_tensor = torch.tensor(U_future_scaled[np.newaxis, :, :], dtype=torch.float32)

            pred_scaled = model(X_tensor, U_past_tensor, U_future_tensor)[0, :, :].numpy()
            pred_real = scaler_output.inverse_transform(pred_scaled.reshape(-1, model.output_size))
            recursive_preds.extend(pred_real)

            # Update states and controls with the last predicted step
            recursive_states = np.vstack([recursive_states[1:], pred_real[-1]])
            recursive_past_controls = np.vstack([recursive_past_controls[1:], U_future[-1]])

        recursive_preds = np.array(recursive_preds)[:total_prediction_length]  # Ensure exact length
        all_recursive_preds.append(recursive_preds)
        all_true_outputs.append(y_test[input_end:pred_end])
        all_time_steps.extend(range(cycle_start - start_idx, pred_end - start_idx))
        all_input_time_steps.extend(range(cycle_start - start_idx, input_end - start_idx))
        all_prediction_time_steps.extend(range(input_end - start_idx, pred_end - start_idx))

all_recursive_preds = np.vstack(all_recursive_preds)
all_true_outputs = np.vstack(all_true_outputs)

# Calculate MSE and R²
mse_pc = mean_squared_error(all_true_outputs[:, 0], all_recursive_preds[:, 0])
r2_pc = r2_score(all_true_outputs[:, 0], all_recursive_preds[:, 0])
mse_Tw_out = mean_squared_error(all_true_outputs[:, 1], all_recursive_preds[:, 1])
r2_Tw_out = r2_score(all_true_outputs[:, 1], all_recursive_preds[:, 1])

print(f"NARX Recursive pc - MSE: {mse_pc:.2f} Bar², R²: {r2_pc:.3f}")
print(f"NARX Recursive Tw_out - MSE: {mse_Tw_out:.2f} K², R²: {r2_Tw_out:.3f}")

# Plot
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
model_pc = []
model_Tw_out = []
for cycle in range(cycles):
    cycle_start = cycle * (sequence_length + total_prediction_length)
    input_end = cycle_start + sequence_length
    pred_end = input_end + total_prediction_length
    model_pc.extend(y_test[cycle_start:input_end, 0])
    model_Tw_out.extend(y_test[cycle_start:input_end, 1] - 273.15)
    model_pc.extend(all_recursive_preds[cycle * total_prediction_length:(cycle + 1) * total_prediction_length, 0])
    model_Tw_out.extend(all_recursive_preds[cycle * total_prediction_length:(cycle + 1) * total_prediction_length, 1] - 273.15)

model_pc = np.array(model_pc)
model_Tw_out = np.array(model_Tw_out)

exp_end = 2680
for cycle in range(cycles):
    cycle_start = cycle * (sequence_length + total_prediction_length)
    input_end = cycle_start + sequence_length
    pred_end = input_end + total_prediction_length
    time_range = np.arange(cycle_start, pred_end)

    ax[0].plot(
        time_range[:sequence_length],
        model_pc[cycle_start:cycle_start + sequence_length],
        color="green", linewidth=3, linestyle="-", marker='o', markersize=5,
        label="Model Input $p_c$" if cycle == 0 else "", zorder=3
    )
    ax[0].plot(
        time_range[sequence_length:],
        model_pc[input_end:pred_end],
        color="orange", linewidth=2, linestyle="-",
        label=f"NARX Predicted $p_c$ (MSE: {mse_pc:.2f} Bar², R²: {r2_pc:.3f})" if cycle == 0 else "", zorder=2
    )

all_true_pc = y_test[start_idx:start_idx + total_steps, 0]
all_time_steps = np.array(all_time_steps)
ax[0].plot(
    all_time_steps[:exp_end],
    all_true_pc[:exp_end],
    label="Experimental $p_c$", color="blue", linestyle="--", linewidth=2.5, marker='x', markersize=4, zorder=1
)
ax[0].plot(
    all_time_steps[exp_end:],
    all_true_pc[exp_end:],
    label="Physical Model $p_c$", color="purple", linestyle="--", linewidth=2.5, marker='x', markersize=4, zorder=1
)

for cycle in range(cycles):
    boundary = cycle * (sequence_length + total_prediction_length) + sequence_length
    ax[0].axvline(x=boundary, color='red', linewidth=2, linestyle=':', label="Input/Prediction Boundary" if cycle == 0 else "")

ax[0].set_ylabel("$p_c$ [Bar]", fontsize=14)
ax[0].legend(loc="upper left", fontsize=12)
ax[0].grid(True)
ax[0].tick_params(axis="both", labelsize=12)

# Water Outlet Temperature
true_output_C = y_test[start_idx:start_idx + total_steps, 1] - 273.15
for cycle in range(cycles):
    cycle_start = cycle * (sequence_length + total_prediction_length)
    input_end = cycle_start + sequence_length
    pred_end = input_end + total_prediction_length
    time_range = np.arange(cycle_start, pred_end)
    ax[1].plot(
        time_range[:sequence_length],
        model_Tw_out[cycle_start:cycle_start + sequence_length],
        color="green", linewidth=3, linestyle="-", marker='o', markersize=5,
        label="Model Input $T_{w,out}$" if cycle == 0 else "", zorder=3
    )
    ax[1].plot(
        time_range[sequence_length:],
        model_Tw_out[input_end:pred_end],
        color="orange", linewidth=2, linestyle="-",
        label=f"NARX Predicted $T_{{w,out}}$ (MSE: {mse_Tw_out:.2f} K², R²: {r2_Tw_out:.3f})" if cycle == 0 else "", zorder=2
    )

ax[1].plot(
    all_time_steps[:exp_end],
    true_output_C[:exp_end],
    label="Experimental $T_{w,out}$", color="blue", linestyle="--", linewidth=2.5, marker='x', markersize=4, zorder=1
)
ax[1].plot(
    all_time_steps[exp_end:],
    true_output_C[exp_end:],
    label="Physical Model $T_{w,out}$", color="purple", linestyle="--", linewidth=2.5, marker='x', markersize=4, zorder=1
)

for cycle in range(cycles):
    boundary = cycle * (sequence_length + total_prediction_length) + sequence_length
    ax[1].axvline(x=boundary, color='red', linewidth=2, linestyle=':', label="Input/Prediction Boundary" if cycle == 0 else "")

ax[1].set_ylabel("$T_{w,out}$ [°C]", fontsize=14)
ax[1].set_xlabel("Time [s]", fontsize=14)
ax[1].legend(loc="upper left", fontsize=12)
ax[1].grid(True)
ax[1].tick_params(axis="both", labelsize=12)

plt.tight_layout()
plt.subplots_adjust(top=0.96, bottom=0.08, left=0.08, right=0.98, hspace=0.25)
plt.show()