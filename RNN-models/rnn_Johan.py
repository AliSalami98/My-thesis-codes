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
X_train = np.column_stack([train_data["Tw_out"]])  # States
U_train = np.column_stack([train_data["omegab"], train_data["Hpev"],
                           train_data["mw_dot"], train_data["Tw_in"]])  # Inputs
y_train = np.column_stack([train_data["Tw_out"]])  # Target (next states)
# y_train = np.column_stack([train_data["Tw_out"]])  # Target (next states)

X_test = np.column_stack([test_data["Tw_out"]])
U_test = np.column_stack([test_data["omegab"], test_data["Hpev"],
                          test_data["mw_dot"], test_data["Tw_in"]])
y_test = np.column_stack([test_data["Tw_out"]])
# y_test = np.column_stack([test_data["Tw_out"]])

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
def create_augmented_sequences(X_full, y_target, seq_len, pred_len):
    past_seq = []
    future_ctrl = []
    target_seq = []

    for i in range(len(X_full) - seq_len - pred_len + 1):
        past = X_full[i:i+seq_len]  # 6 features
        future = X_full[i+seq_len:i+seq_len+pred_len, 1:]  # control = 4 features
        target = y_target[i+seq_len:i+seq_len+pred_len]  # prediction targets

        past_seq.append(past)
        future_ctrl.append(future)
        target_seq.append(target)

    return np.array(past_seq), np.array(future_ctrl), np.array(target_seq)


# Parameters
sequence_length = 10 # Number of time steps per sequence
prediction_length = 10  # Number of steps to predict ahead
batch_size = 32
use_batch = True  # Toggle batch processing on (True) or off (False)

# Prepare training, validation, and test sequences
X_train_seq, U_train_seq, y_train_seq = create_augmented_sequences(input_train_split, y_train_split, sequence_length, prediction_length)
X_val_seq, U_val_seq, y_val_seq = create_augmented_sequences(input_val, y_val, sequence_length, prediction_length)
X_test_seq, U_test_seq, y_test_seq = create_augmented_sequences(input_test, y_test, sequence_length, prediction_length)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
U_train_tensor = torch.tensor(U_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
U_val_tensor = torch.tensor(U_val_seq, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
U_test_tensor = torch.tensor(U_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

# Prepare DataLoaders for batch processing (disable shuffling)
# Prepare DataLoaders for batch processing (disable shuffling)
train_dataset = TensorDataset(X_train_tensor, U_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, U_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, U_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class RNNCellModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNCellModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNNCell stack
        self.rnn_cells = nn.ModuleList([
            nn.RNNCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, past_inputs, future_controls):
        # past_inputs: (batch, seq_len, 6)
        # future_controls: (batch, pred_len, 4)
        batch_size = past_inputs.size(0)
        seq_len = past_inputs.size(1)
        pred_len = future_controls.size(1)

        # Initialize hidden states (no cell state in RNNCell)
        h = [torch.zeros(batch_size, self.hidden_size, device=past_inputs.device) for _ in range(self.num_layers)]

        # --------- Encode past sequence ---------
        for t in range(seq_len):
            x_t = past_inputs[:, t, :]
            for i, cell in enumerate(self.rnn_cells):
                h[i] = cell(x_t, h[i])
                x_t = h[i]

        # Get the last output state (pc, Tw_out)
        last_output = past_inputs[:, -1, :1]  # shape: (batch, 2)

        # --------- Decode future steps ---------
        predictions = []
        for t in range(pred_len):
            control_t = future_controls[:, t, :]  # shape: (batch, 4)
            input_t = torch.cat([last_output, control_t], dim=-1)  # shape: (batch, 6)

            for i, cell in enumerate(self.rnn_cells):
                h[i] = cell(input_t, h[i])
                input_t = h[i]

            out = self.fc(h[-1])  # shape: (batch, 2)
            predictions.append(out.unsqueeze(1))  # keep time dimension
            last_output = out  # Feed prediction back

        return torch.cat(predictions, dim=1)  # shape: (batch, pred_len, 2)



# Hyperparameters
input_size = 5    # Number of input features (state + inputs)
hidden_size = 16 # RNN hidden layer size
output_size = 1   # Number of output features per time step (pc, Tw_out)
num_layers = 1    # Number of RNN layers
num_epochs = 150
learning_rate = 0.001
dropout = 0.1

# Initialize RNN model, loss function, and optimizer
model = RNNCellModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

import time
start_time = time.time()

# ------------------------
# Training Loop with Early Stopping
# ------------------------
best_val_loss = float('inf')
patience = 30  # Number of epochs to wait for improvement
patience_counter = 0
train_losses = []
val_losses = []
min_delta = 1e-6

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    if use_batch:  # Batch processing
        for X_train_batch, U_train_batch, y_train_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_train_batch, U_train_batch)
            loss = criterion(outputs, y_train_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_train_batch.size(0)
        train_loss = total_train_loss / len(train_dataset)
    else:  # Non-batch processing
        optimizer.zero_grad()
        outputs = model(X_train_tensor, U_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

    # Validation loss
    model.eval()
    with torch.no_grad():
        if use_batch:  # Batch processing for validation
            total_val_loss = 0.0
            for X_val_batch, U_val_batch, y_val_batch in val_loader:
                val_outputs = model(X_val_batch, U_val_batch)
                val_loss = criterion(val_outputs, y_val_batch)
                total_val_loss += val_loss.item() * X_val_batch.size(0)
            val_loss = total_val_loss / len(val_dataset)
        else:  # Non-batch processing for validation
            val_outputs = model(X_val_tensor, U_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

    # Store losses for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_loss_denorm = val_loss * (scaler_output.scale_ ** 2).mean()
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_RNN_model.pth")  # Save best model
    # else:
    #     patience_counter += 1
    #     if patience_counter >= patience:
    #         print(f"Early stopping triggered after epoch {epoch + 1}")
    #         break

    scheduler.step(val_loss)  # Adjust learning rate based on validation loss

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
# Load the best model
model.load_state_dict(torch.load("best_RNN_model.pth"))

# ------------------------
# Professional Plot: Training & Validation Loss
# ------------------------
import matplotlib.ticker as ticker

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss", linewidth=2.5, color="steelblue")
plt.plot(val_losses, label="Validation Loss", linewidth=2.5, color="darkorange", linestyle="--")

plt.xlabel("Epoch", fontsize=14)
plt.ylabel("MSE Loss", fontsize=14)
# plt.title("Training and Validation Loss", fontsize=16)
plt.legend(loc="upper right", fontsize=12, frameon=False)

# Improve tick formatting
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # integer ticks for epochs
plt.grid(True, which='major', linestyle=':', linewidth=1, alpha=0.7)

# Tight layout for articles
plt.tight_layout()

plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\control\RNN_training.eps", format='eps', bbox_inches='tight')

plt.show()
# ------------------------
# Evaluation Loop
# ------------------------
model.eval()
with torch.no_grad():
    train_predictions = model(X_train_tensor, U_train_tensor)
    train_loss = criterion(train_predictions, y_train_tensor).item()

    val_predictions = model(X_val_tensor, U_val_tensor)
    val_loss = criterion(val_predictions, y_val_tensor).item()

    test_predictions = model(X_test_tensor, U_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor).item()

print(f"Training MSE: {train_loss:.4f}")
print(f"Validation MSE: {val_loss:.4f}")
print(f"Test MSE: {test_loss:.4f}")

# ------------------------
# Save the Model and Scalers
# ------------------------
torch.save(model.state_dict(), "RNN_Johan3/rnn_model.pth")
joblib.dump(scaler_input, "RNN_Johan3/scaler_input.pkl")
joblib.dump(scaler_output, "RNN_Johan3/scaler_output.pkl")

print("Model and scalers saved.")

with torch.no_grad():
    test_predictions = model(X_test_tensor, U_test_tensor)
    
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


mse_Tw_out, mape_Tw_out, r2_Tw_out = calculate_metrics(y_test_denorm[:, 0], test_predictions_denorm[:, 0])
# mse_Tw_out, mape_Tw_out, r2_Tw_out = calculate_metrics(y_test_denorm[:, 1], test_predictions_denorm[:, 1])

# ------------------------
# Plot
# ------------------------
fig, ax = plt.subplots(1, 1, figsize=(14, 10), sharex=True)

y_true_C = y_test_denorm[:, 0] - 273.15
y_pred_C = test_predictions_denorm[:, 0] - 273.15
# Plot Water Outlet Temperature
ax.plot(y_true_C, label="Real + physical model data", color="blue", linewidth=2)
ax.plot(y_pred_C,
           label=f"RNN Prediction (MSE: {mse_Tw_out:.2f} K², R²: {r2_Tw_out:.3f})",
           linestyle="dashed", color="orange", linewidth=2)
ax.set_xlabel("Time [s]", fontsize=14)
ax.set_ylabel("$T_\text{w, out}$ [°C]", fontsize=14)
ax.legend(loc="upper left", fontsize=12)
ax.tick_params(axis="both", labelsize=12)
ax.grid(True)

plt.tight_layout()
plt.show()



model = RNNCellModel(input_size, hidden_size, output_size, num_layers)
model.load_state_dict(torch.load("RNN_Johan3/rnn_model.pth"))
model.eval()

scaler_input = joblib.load("RNN_Johan3/scaler_input.pkl")
scaler_output = joblib.load("RNN_Johan3/scaler_output.pkl")
model.eval()
with torch.no_grad():
    val_predictions = model(X_val_tensor, U_val_tensor)
    y_val_np = y_val_tensor.numpy()
    val_predictions_np = val_predictions.numpy()

    # Flatten all time steps
    y_val_np_flat = y_val_np.reshape(-1, 1)  # shape: (batch * pred_len, 1)
    val_predictions_np_flat = val_predictions_np.reshape(-1, 1)

    # Denormalize
    y_val_denorm = scaler_output.inverse_transform(y_val_np_flat)
    val_predictions_denorm = scaler_output.inverse_transform(val_predictions_np_flat)

    # Calculate denormalized MSE
    val_loss_denorm = mean_squared_error(y_val_denorm, val_predictions_denorm)
    r2_val_denorm = r2_score(y_val_denorm, val_predictions_denorm)
    print(f"Denormalized Validation MSE: {val_loss_denorm:.4f} K²")
    print(f"Denormalized Validation R²: {r2_val_denorm:.3f}")

# Recursion

# Parameters
prediction_length = 500  # Steps to predict per cycle
total_steps = 10 *(sequence_length + prediction_length)  # Total simulation length
cycles = total_steps // (sequence_length + prediction_length)  # Number of cycles (e.g., 5 for 200 steps)

# Prepare the full input and normalize
X_test = np.column_stack([test_data["Tw_out"]])
U_test = np.column_stack([test_data["omegab"], test_data["Hpev"], test_data["mw_dot"], test_data["Tw_in"]])
y_test_true = np.column_stack([test_data["Tw_out"]])
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

        # Get input sequence and known future controls
        recursive_input = input_test_scaled[cycle_start:input_end].copy()  # shape: (seq_len, 6)
        future_controls = input_test_scaled[input_end:pred_end, 1:]        # shape: (pred_len, 4)

        # Reshape to batch of 1
        recursive_input_tensor = torch.tensor(recursive_input[np.newaxis, :, :], dtype=torch.float32)  # (1, seq_len, 6)
        future_controls_tensor = torch.tensor(future_controls[np.newaxis, :, :], dtype=torch.float32)  # (1, pred_len, 4)

        # Predict using the model
        pred_scaled = model(recursive_input_tensor, future_controls_tensor).squeeze(0).numpy()  # shape: (pred_len, 2)

        # Inverse transform predictions to real values
        pred_real = scaler_output.inverse_transform(pred_scaled)  # shape: (pred_len, 2)

        # Store predictions and true outputs
        all_recursive_preds.append(pred_real)
        all_true_outputs.append(y_test_true[input_end:pred_end])
        all_time_steps.extend(range(cycle_start - start_idx, pred_end - start_idx))
        all_input_time_steps.extend(range(cycle_start - start_idx, input_end - start_idx))
        all_prediction_time_steps.extend(range(input_end - start_idx, pred_end - start_idx))


# Convert to arrays
all_recursive_preds = np.vstack(all_recursive_preds)  # shape: (100, 2) for 5 cycles
all_true_outputs = np.vstack(all_true_outputs)  # shape: (100, 2)

# Calculate MSE and R²
mse_Tw_out = mean_squared_error(all_true_outputs[:, 0], all_recursive_preds[:, 0])
r2_Tw_out = r2_score(all_true_outputs[:, 0], all_recursive_preds[:, 0])

# Print results
print(f"Alternating Recursive Tw_out - MSE: {mse_Tw_out:.2f} K², R²: {r2_Tw_out:.3f}")


# Combine input and predicted values for pc and Tw_out
model_pc = []
model_Tw_out = []
for cycle in range(cycles):
    cycle_start = start_idx + cycle * (sequence_length + prediction_length)
    input_end = cycle_start + sequence_length
    # Append input values (from y_test_true)
    model_Tw_out.extend(y_test_true[cycle_start:input_end, 0] - 273.15)  # Convert to Celsius
    # Append predicted values
    model_Tw_out.extend(all_recursive_preds[cycle * prediction_length:(cycle + 1) * prediction_length, 0] - 273.15)  # Convert to Celsius

model_pc = np.array(model_pc)
model_Tw_out = np.array(model_Tw_out)

# Plot setup
fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharex=True)

exp_end = 2680

# -------- Water Outlet Temperature: $T_\text{w, sup}$ --------
true_output_C = y_test_true[start_idx:start_idx + total_steps, 0] - 273.15

for cycle in range(cycles):
    cycle_start = cycle * (sequence_length + prediction_length)
    input_end = cycle_start + sequence_length
    pred_end = cycle_start + sequence_length + prediction_length

    # Input segment (green) — emphasized
    ax.plot(
        all_time_steps[cycle_start:input_end],
        model_Tw_out[cycle_start:input_end],
        color="green", linewidth=3, linestyle="-", marker='o', markersize=5,
        label=r"RNN input $T_\text{w, sup}$" if cycle == 0 else "", zorder=3
    )

    # Predicted segment (orange)
    ax.plot(
        all_time_steps[input_end:pred_end],
        model_Tw_out[input_end:pred_end],
        color="orange", linewidth=2, linestyle="--",
        label=rf"RNN predicted $T_{{\mathrm{{w,\,sup}}}}$ (MSE: {mse_Tw_out:.2f} $\mathrm{{K}}^2$, R$^2$: {r2_Tw_out:.3f})" if cycle == 0 else "", zorder=2
    )

# True data (blue dashed)
# Separate true data
true_output_C = y_test_true[start_idx:start_idx + total_steps, 0] - 273.15

# Plot experimental
ax.plot(
    all_time_steps[:exp_end],
    true_output_C[:exp_end],
    label=r"Experimental $T_\text{w, sup}$", color="blue", linestyle="-", linewidth=2.5, zorder=1
)

# Plot physical model
ax.plot(
    all_time_steps[exp_end:],
    true_output_C[exp_end:],
    label=r"Reference model $T_\text{w, sup}$", color="purple", linestyle="-", linewidth=2.5, zorder=1
)

for cycle in range(cycles):
    boundary = (cycle + 1) * sequence_length + cycle * prediction_length
    ax.axvline(x=boundary, color='k', linewidth=2, linestyle=':', label="Input/prediction boundary" if cycle == 0 else "")

ax.set_ylabel(r"Supply water temperature $T_\text{w, sup}$ [°C]", fontsize=16)
ax.set_xlabel("Time [s]", fontsize=16)
ax.legend(loc="upper left", fontsize=14)
# ax.grid(True)
ax.tick_params(axis="both", labelsize=14)
# Optimize spacing
plt.tight_layout()  # Automatically adjust subplots to fit figure area
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\control\RNN_validation.eps", format='eps', bbox_inches='tight')


plt.show()
