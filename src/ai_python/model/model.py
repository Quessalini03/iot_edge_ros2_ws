import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Load data
X = np.load("sensor_sequences.npy")  # shape: (samples, 5, 2)
y = np.load("labels.npy")            # shape: (samples,)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Create Dataset & Dataloader
dataset = TensorDataset(X_tensor, y_tensor)

# Train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Define LSTM model
class FireLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1):
        super(FireLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # binary classification (fire or no fire)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the last output in the sequence
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Model, loss, optimizer
model = FireLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} - Val Acc: {100 * correct / total:.2f}%")

# Save model checkpoint
torch.save(model.state_dict(), "fire_lstm_checkpoint.pth")
print("✅ Model saved to fire_lstm_checkpoint.pth")
