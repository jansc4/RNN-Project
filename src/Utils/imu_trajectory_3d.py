import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from mpl_toolkits.mplot3d import Axes3D

# =====================================================
# 1. Filtr Butterwortha
# =====================================================
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

# =====================================================
# 2. Wczytanie danych
# =====================================================
df = pd.read_csv("mpu_data6.csv")  # zmień nazwę pliku

time = df["timestamp"].values
dt = np.mean(np.diff(time))
fs = 1.0 / dt
print(f"fs ≈ {fs:.2f} Hz")

# =====================================================
# 3. Filtracja przyspieszeń
# =====================================================
cutoff = 10  # Hz
ax = butter_lowpass_filter(df["ax"], cutoff, fs)
ay = butter_lowpass_filter(df["ay"], cutoff, fs)
az = butter_lowpass_filter(df["az"], cutoff, fs)

# =====================================================
# 4. Usunięcie grawitacji – prosta metoda
# =====================================================
# Możesz zastąpić to filtrem komplementarnym lub orientacją z żyroskopu
az = az - 0.95  # zakładamy że spoczynek → az ≈ 1g

# =====================================================
# 5. Integracja: acc → vel → pos
# =====================================================
vx = np.cumsum(ax * dt)
vy = np.cumsum(ay * dt)
vz = np.cumsum(az * dt)

px = np.cumsum(vx * dt)
py = np.cumsum(vy * dt)
pz = np.cumsum(vz * dt)

# =====================================================
# 6. Trajektoria 3D
# =====================================================
fig = plt.figure(figsize=(10, 7))
ax3d = fig.add_subplot(111, projection='3d')

ax3d.plot(px, py, pz, linewidth=2)

ax3d.set_title("Trajektoria 3D z akcelerometru")
ax3d.set_xlabel("X [m]")
ax3d.set_ylabel("Y [m]")
ax3d.set_zlabel("Z [m]")

# odwrócenie osi Z jeśli chcesz widok “w dół”
# ax3d.invert_zaxis()

plt.tight_layout()
plt.show()
