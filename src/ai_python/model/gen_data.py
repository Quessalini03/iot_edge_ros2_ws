import numpy as np
import pandas as pd
import random

# Configuration
num_sequences = 10000
sequence_length = 5
fire_ratio = 0.1  # 10% of data sequences are fire events

def generate_normal_sequence():
    temp = np.clip(np.random.normal(22, 3, sequence_length), -40, 80)
    hum = np.clip(np.random.normal(50, 10, sequence_length), 0, 100)
    return np.stack([temp, hum], axis=1)

def generate_fire_sequence():
    # Simulate a fire event over the sequence (e.g., time step 3 to 5)
    base_temp = np.random.normal(22, 2)
    base_hum = np.random.normal(50, 5)

    temp = np.array([base_temp + i*10 if i >= 2 else base_temp for i in range(sequence_length)])
    temp = np.clip(temp + np.random.normal(0, 2, sequence_length), -40, 80)

    hum = np.array([base_hum - i*10 if i >= 2 else base_hum for i in range(sequence_length)])
    hum = np.clip(hum + np.random.normal(0, 3, sequence_length), 0, 100)

    return np.stack([temp, hum], axis=1)

# Generate dataset
data = []
labels = []

for _ in range(num_sequences):
    if random.random() < fire_ratio:
        seq = generate_fire_sequence()
        label = 1
    else:
        seq = generate_normal_sequence()
        label = 0
    data.append(seq)
    labels.append(label)

# Save as npy files for PyTorch training
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int64)

np.save("sensor_sequences.npy", data)
np.save("labels.npy", labels)

print(f"Generated {data.shape[0]} sequences. Shape: {data.shape} (samples, timesteps, features)")
