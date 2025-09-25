import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Theater2_data import train_data, test_data
import joblib

# ------------------------
# Prepare Data
# ------------------------
# Combine train_data into input and target arrays
X_train = np.column_stack([train_data["Theater2"]])  # States
U_train = np.column_stack([train_data["omegab"], train_data["omega2"]])  # Inputs

X_test = np.column_stack([test_data["Theater2"]])
U_test = np.column_stack([test_data["omegab"], test_data["omega2"]])

from sklearn.metrics import mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from Theater2_data import train_data, test_data

# ------------------------
# Prepare Data
# ------------------------
# Combine inputs and targets
X_train = np.column_stack([train_data["Theater2"]])  # States
U_train = np.column_stack([train_data["omegab"], train_data["omega2"]])  # Inputs
X_test = np.column_stack([test_data["Theater2"]])
U_test = np.column_stack([test_data["omegab"], test_data["omega2"]])

# Combine state and input for training and testing
train_x = np.hstack([X_train, U_train])
train_y = X_train  # Target: next Theater2
test_x = np.hstack([X_test, U_test])
test_y = X_test  # Target: next Theater2

# Convert to PyTorch tensors
train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
test_y_tensor = torch.tensor(test_y, dtype=torch.float32)

# ------------------------
# Define Neural Network
# ------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Hyperparameters
input_size = train_x.shape[1]
hidden_size = 32
output_size = train_y.shape[1]
learning_rate = 0.01
num_epochs = 10
batch_size = 4

# Initialize model, loss function, and optimizer
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ------------------------
# Training Loop
# ------------------------
train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# Save the trained model
model_save_path = "simple_nn_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# ------------------------
# Validation
# ------------------------
model.eval()
with torch.no_grad():
    val_pred_tensor = model(test_x_tensor)
    val_pred = val_pred_tensor.numpy()
    val_y = test_y_tensor.numpy()

# Calculate metrics
mape_val = mean_absolute_percentage_error(val_y.flatten(), val_pred.flatten()) * 100
r2_val = r2_score(val_y.flatten(), val_pred.flatten())
print(f"Validation MAPE: {mape_val:.2f}%")
print(f"Validation R²: {r2_val:.2f}")

# ------------------------
# Plot Results
# ------------------------
plt.figure(figsize=(10, 6))
plt.plot(val_y.flatten(), label="Observed", color="blue")
plt.plot(val_pred.flatten(), label="Predicted", linestyle="dashed", color="orange")
plt.legend()
plt.xlabel("Samples")
plt.ylabel("Theater2")
plt.title("Observed vs Predicted")
plt.grid()
plt.show()
