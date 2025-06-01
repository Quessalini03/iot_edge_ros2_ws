import numpy as np
import matplotlib.pyplot as plt

# Load the generated data
data = np.load("sensor_sequences.npy")
labels = np.load("labels.npy")

# Separate normal and fire sequences
normal_indices = np.where(labels == 0)[0]
fire_indices = np.where(labels == 1)[0]

# Plot a few sequences
def plot_sequences(indices, title):
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(indices[:5]):
        seq = data[idx]
        temp = seq[:, 0]
        hum = seq[:, 1]
        plt.subplot(2, 5, i + 1)
        plt.plot(temp, label="Temp (°C)", color='r')
        plt.plot(hum, label="Humidity (%)", color='b')
        plt.title(f"{title} #{i+1}")
        plt.ylim([-10, 100])
        plt.grid(True)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()

print("Normal Sequences:")
plot_sequences(normal_indices, "Normal")

print("Fire Sequences:")
plot_sequences(fire_indices, "Fire")
