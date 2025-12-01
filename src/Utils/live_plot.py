import serial
import struct
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

class LiveMPU6050Viewer:
    def __init__(self, port='COM3', baudrate=921600, window_size=500):
        """
        Live viewer dla danych MPU6050
        
        Args:
            port: Port szeregowy
            baudrate: Prędkość transmisji
            window_size: Liczba próbek w oknie (historia)
        """
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        
        self.window_size = window_size
        self.running = False
        self.measuring = False
        
        # Bufory danych (deque = szybka kolejka FIFO)
        self.timestamps = deque(maxlen=window_size)
        self.ax_data = deque(maxlen=window_size)
        self.ay_data = deque(maxlen=window_size)
        self.az_data = deque(maxlen=window_size)
        self.gx_data = deque(maxlen=window_size)
        self.gy_data = deque(maxlen=window_size)
        self.gz_data = deque(maxlen=window_size)
        
        # Skale konwersji
        self.accel_scale = 16384.0
        self.gyro_scale = 131.0
        
        # Statystyki
        self.sample_count = 0
        self.start_time = None
        self.last_timestamp = 0
        
        # Czekaj na "READY"
        while True:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if line == "READY":
                print("✓ ESP32 gotowy")
                break
        
        # Wątek do odczytu danych
        self.read_thread = None
    
    def start_measurement(self):
        """Rozpocznij pomiary"""
        self.ser.reset_input_buffer()
        self.ser.write(b'S')
        time.sleep(0.1)
        
        self.measuring = True
        self.sample_count = 0
        self.start_time = time.time()
        
        print("✓ Pomiary rozpoczęte")
    
    def stop_measurement(self):
        """Zatrzymaj pomiary"""
        self.measuring = False
        time.sleep(0.1)
        
        self.ser.reset_input_buffer()
        self.ser.write(b'X')
        
        print("✓ Pomiary zatrzymane")
    
    def read_data_thread(self):
        """Wątek odczytujący dane z Serial"""
        while self.running:
            if self.measuring and self.ser.in_waiting >= 19:
                try:
                    # Szukaj markera 'D'
                    byte = self.ser.read(1)
                    if byte != b'D':
                        continue
                    
                    # Odczytaj timestamp
                    timestamp_bytes = self.ser.read(4)
                    if len(timestamp_bytes) != 4:
                        continue
                    timestamp = struct.unpack('<I', timestamp_bytes)[0] / 1e6
                    
                    # Odczytaj dane
                    data_bytes = self.ser.read(14)
                    if len(data_bytes) != 14:
                        continue
                    
                    raw_data = struct.unpack('>7h', data_bytes)
                    
                    # Dodaj do buforów
                    self.timestamps.append(timestamp)
                    self.ax_data.append(raw_data[0] / self.accel_scale)
                    self.ay_data.append(raw_data[1] / self.accel_scale)
                    self.az_data.append(raw_data[2] / self.accel_scale)
                    self.gx_data.append(raw_data[4] / self.gyro_scale)
                    self.gy_data.append(raw_data[5] / self.gyro_scale)
                    self.gz_data.append(raw_data[6] / self.gyro_scale)
                    
                    self.sample_count += 1
                    self.last_timestamp = timestamp
                    
                except Exception as e:
                    continue
            else:
                time.sleep(0.001)  # Krótka pauza żeby nie spamować CPU
    
    def get_sampling_rate(self):
        """Oblicz aktualną częstotliwość próbkowania"""
        if self.start_time is None or self.sample_count < 10:
            return 0
        
        elapsed = time.time() - self.start_time
        return self.sample_count / elapsed if elapsed > 0 else 0
    
    def init_plot(self):
        """Inicjalizacja wykresów"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Akcelerometr
        self.line_ax, = self.ax1.plot([], [], 'r-', label='X', linewidth=1.5)
        self.line_ay, = self.ax1.plot([], [], 'g-', label='Y', linewidth=1.5)
        self.line_az, = self.ax1.plot([], [], 'b-', label='Z', linewidth=1.5)
        
        self.ax1.set_xlim(0, self.window_size * 0.01)  # Początkowe okno
        self.ax1.set_ylim(-3, 3)
        self.ax1.set_ylabel('Przyspieszenie [g]', fontsize=11)
        self.ax1.set_title('Akcelerometr - Live', fontsize=13, fontweight='bold')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)
        
        # Żyroskop
        self.line_gx, = self.ax2.plot([], [], 'r-', label='X', linewidth=1.5)
        self.line_gy, = self.ax2.plot([], [], 'g-', label='Y', linewidth=1.5)
        self.line_gz, = self.ax2.plot([], [], 'b-', label='Z', linewidth=1.5)
        
        self.ax2.set_xlim(0, self.window_size * 0.01)
        self.ax2.set_ylim(-250, 250)
        self.ax2.set_xlabel('Czas [s]', fontsize=11)
        self.ax2.set_ylabel('Prędkość kątowa [°/s]', fontsize=11)
        self.ax2.set_title('Żyroskop - Live', fontsize=13, fontweight='bold')
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Tekst z statystykami
        self.stats_text = self.fig.text(0.02, 0.98, '', fontsize=10, 
                                        verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def update_plot(self, frame):
        """Aktualizacja wykresu (wywoływana przez FuncAnimation)"""
        if len(self.timestamps) < 2:
            return self.line_ax, self.line_ay, self.line_az, self.line_gx, self.line_gy, self.line_gz
        
        # Konwertuj deque na numpy array (szybsze)
        t = np.array(self.timestamps)
        t = t - t[0]  # Normalizuj do zera
        
        # Aktualizuj dane linii
        self.line_ax.set_data(t, np.array(self.ax_data))
        self.line_ay.set_data(t, np.array(self.ay_data))
        self.line_az.set_data(t, np.array(self.az_data))
        
        self.line_gx.set_data(t, np.array(self.gx_data))
        self.line_gy.set_data(t, np.array(self.gy_data))
        self.line_gz.set_data(t, np.array(self.gz_data))
        
        # Automatyczne skalowanie osi X
        if len(t) > 0:
            self.ax1.set_xlim(max(0, t[-1] - 5), t[-1] + 0.5)  # 5s okno
            self.ax2.set_xlim(max(0, t[-1] - 5), t[-1] + 0.5)
        
        # Automatyczne skalowanie osi Y dla akcelerometru
        if len(self.ax_data) > 0:
            all_accel = list(self.ax_data) + list(self.ay_data) + list(self.az_data)
            margin = 0.5
            self.ax1.set_ylim(min(all_accel) - margin, max(all_accel) + margin)
        
        # Automatyczne skalowanie osi Y dla żyroskopu
        if len(self.gx_data) > 0:
            all_gyro = list(self.gx_data) + list(self.gy_data) + list(self.gz_data)
            margin = 20
            self.ax2.set_ylim(min(all_gyro) - margin, max(all_gyro) + margin)
        
        # Aktualizuj statystyki
        fps = self.get_sampling_rate()
        stats = f'Próbek: {self.sample_count} | Częstotliwość: {fps:.1f} Hz | Czas: {self.last_timestamp:.2f}s'
        self.stats_text.set_text(stats)
        
        return self.line_ax, self.line_ay, self.line_az, self.line_gx, self.line_gy, self.line_gz
    
    def run(self):
        """Uruchom live viewer"""
        self.running = True
        
        # Start wątku odczytu
        self.read_thread = threading.Thread(target=self.read_data_thread, daemon=True)
        self.read_thread.start()
        
        # Start pomiarów
        self.start_measurement()
        
        # Inicjalizacja wykresu
        self.init_plot()
        
        # Animacja (odświeżanie co 50ms = 20 FPS)
        self.anim = FuncAnimation(self.fig, self.update_plot, 
                                 interval=50, blit=True, cache_frame_data=False)
        
        print("✓ Live viewer uruchomiony. Zamknij okno aby zatrzymać.")
        plt.show()
        
        # Cleanup po zamknięciu okna
        self.running = False
        self.stop_measurement()
        self.ser.close()
        print("✓ Zakończono")


# Przykładowe użycie
if __name__ == "__main__":
    viewer = LiveMPU6050Viewer(port='COM12', window_size=1000)  # 1000 próbek w oknie
    viewer.run()