import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_sequences(data, window_size=200):
    data = np.array(data)
    N, W, F = data.shape

    data_2d = data.reshape(-1, F)
    scaler = StandardScaler()
    data_scaled_2d = scaler.fit_transform(data_2d)

    data_scaled = data_scaled_2d.reshape(N, W, F)
    return data_scaled.astype(np.float32), scaler