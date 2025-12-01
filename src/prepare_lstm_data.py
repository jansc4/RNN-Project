# prepare_lstm_data.py
import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
RAW_TRAIN_DIR = "../data/training_data_raw/"
RAW_ANOMALY_DIR = "../data/anomaly_data_raw/"
REF_TRAIN_DIR = "../data/training_data_refined/"
REF_ANOMALY_DIR = "../data/anomaly_data_refined/"

WINDOW_SIZE = 200
CUT_OFF = 10
USE_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]

# -------------------------------------------------------
# 1. Filtr dolnoprzepustowy
# -------------------------------------------------------
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

# -------------------------------------------------------
# 2. Przetwarzanie pojedynczego pliku CSV → .NPY
# -------------------------------------------------------
def process_file(path, output_dir):
    df = pd.read_csv(path)

    # Obliczenie częstotliwości próbkowania
    time = df["timestamp"].values
    dt = np.mean(np.diff(time))
    fs = 1.0 / dt

    # Filtracja IMU
    for col in USE_COLUMNS:
        df[col] = butter_lowpass_filter(df[col].values, CUT_OFF, fs)

    # wybór IMU
    data = df[USE_COLUMNS].values

    # Cięcie na sekwencje
    total_len = len(data)
    num_windows = total_len // WINDOW_SIZE

    sequences = []
    for i in range(num_windows):
        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE
        sequences.append(data[start:end])

    sequences = np.array(sequences)

    # zapis
    filename = os.path.basename(path).replace(".csv", ".npy")
    np.save(os.path.join(output_dir, filename), sequences)

    print(f"✓ Zapisano: {filename}  ({sequences.shape})")

# -------------------------------------------------------
# 3. Przetwarzanie całego katalogu
# -------------------------------------------------------
def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    print(f"\n### Przetwarzanie katalogu: {input_dir} ({len(files)} plików)")
    
    for f in files:
        process_file(os.path.join(input_dir, f), output_dir)

# -------------------------------------------------------
# 4. Main
# -------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("PRZYGOTOWANIE DANYCH LSTM")
    print("="*60)
    
    # Przetwarzanie danych treningowych
    if os.path.exists(RAW_TRAIN_DIR):
        process_directory(RAW_TRAIN_DIR, REF_TRAIN_DIR)
    else:
        print(f"⚠ Brak katalogu: {RAW_TRAIN_DIR}")
    
    # Przetwarzanie danych anomalii
    if os.path.exists(RAW_ANOMALY_DIR):
        process_directory(RAW_ANOMALY_DIR, REF_ANOMALY_DIR)
    else:
        print(f"⚠ Brak katalogu: {RAW_ANOMALY_DIR}")
    
    print("\n✓ Zakończono!")