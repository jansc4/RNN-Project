import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ================================================================
# 1. FUNKCJE FILTRUJĄCE
# ================================================================
def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Filtr dolnoprzepustowy Butterwortha.
    cutoff - częstotliwość odcięcia (Hz)
    fs - częstotliwość próbkowania (Hz)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


# ================================================================
# 2. WCZYTANIE DANYCH Z CSV
# ================================================================
df = pd.read_csv("mpu_data6.csv")    # <-- zmień nazwę pliku jeśli potrzeba

# czas może nie być równy — więc policzymy fs jako średnią
time = df["timestamp"].values
dt = np.mean(np.diff(time))
fs = 1.0 / dt

print(f"Częstotliwość próbkowania ~ {fs:.2f} Hz")

# ================================================================
# 3. FILTROWANIE DANYCH
# ================================================================
cutoff = 10  # Hz – zwykle 5–20 Hz odpowiednie do filtracji IMU

filtered = {}
for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
    filtered[col] = butter_lowpass_filter(df[col].values, cutoff, fs)

# ================================================================
# 4. RYSOWANIE WYKRESÓW PRZED I PO
# ================================================================
plt.figure(figsize=(12, 8))

signals = ["ax", "ay", "az", "gx", "gy", "gz"]

for i, sig in enumerate(signals, 1):
    plt.subplot(3, 2, i)
    plt.plot(time, df[sig], label=f"{sig} raw", alpha=0.6)
    plt.plot(time, filtered[sig], label=f"{sig} filtered", linewidth=2)
    plt.title(sig)
    plt.xlabel("time [s]")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
