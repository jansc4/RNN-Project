import serial
import serial.tools.list_ports
import struct
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def find_esp32_port():
    """
    Znajdź port ESP32 automatycznie
    
    Returns:
        str: Nazwa portu lub None
    """
    ports = serial.tools.list_ports.comports()
    
    print("Dostępne porty:")
    for i, port in enumerate(ports):
        print(f"  [{i}] {port.device} - {port.description}")
        # ESP32 często zawiera "USB", "UART", "CP210", "CH340" w opisie
        if any(keyword in port.description.upper() for keyword in ['USB', 'UART', 'CP210', 'CH340', 'SERIAL']):
            print(f"      ^ Prawdopodobnie ESP32")
    
    if len(ports) == 0:
        print("✗ Nie znaleziono żadnych portów szeregowych")
        return None
    elif len(ports) == 1:
        print(f"\n✓ Wykryto jeden port: {ports[0].device}")
        return ports[0].device
    else:
        print("\nWybierz numer portu lub naciśnij Enter dla automatycznego wykrywania:")
        choice = input("> ").strip()
        if choice.isdigit() and int(choice) < len(ports):
            return ports[int(choice)].device
        else:
            # Spróbuj znaleźć ESP32
            for port in ports:
                if any(keyword in port.description.upper() for keyword in ['USB', 'CP210', 'CH340']):
                    return port.device
            return ports[0].device

def check_port_available(port):
    """
    Sprawdź czy port jest dostępny
    
    Returns:
        bool: True jeśli port jest dostępny
    """
    try:
        s = serial.Serial(port)
        s.close()
        time.sleep(0.5)
        return True
    except serial.SerialException:
        return False
    except Exception as e:
        print(f"Błąd sprawdzania portu: {e}")
        return False

class MPU6050Controller:
    def __init__(self, port='COM3', baudrate=921600):
        """
        Inicjalizacja połączenia z ESP32
        
        Args:
            port: Port szeregowy (np. 'COM3' na Windows, '/dev/ttyUSB0' na Linux)
            baudrate: Prędkość transmisji
        """
        self.ser = serial.Serial(port, baudrate, timeout=1)
        
        # KLUCZOWE: Reset ESP32 przez DTR
        self.ser.setDTR(False)
        time.sleep(0.1)
        self.ser.setDTR(True)
        time.sleep(2)  # Czekaj na restart ESP32
        
        # Bufor na dane
        self.data = []
        self.running = False
        
        # Konwersja z raw na jednostki fizyczne
        self.accel_scale = 16384.0  # LSB/g dla ±2g
        self.gyro_scale = 131.0     # LSB/(°/s) dla ±250°/s
        
        # Wyczyść stare dane z bufora
        self.ser.reset_input_buffer()
        
        # Czekaj na "READY"
        timeout = 3
        start = time.time()
        while True:
            if time.time() - start > timeout:
                print("✗ ESP32 nie wysłał READY - kontynuuję mimo to")
                break
            
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line == "READY":
                    print("✓ ESP32 gotowy")
                    break
    
    def start_measurement(self):
        """Rozpocznij pomiary"""
        self.ser.write(b'S')
        response = self.ser.readline().decode('utf-8').strip()
        if response == "OK:START":
            print("✓ Pomiary rozpoczęte")
            self.running = True
            self.data = []
        else:
            print(f"✗ Błąd: {response}")
    
    def stop_measurement(self):
        """Zatrzymaj pomiary"""
        self.running = False
        time.sleep(0.1)  # Poczekaj na ostatnie próbki
        
        # Wyczyść bufor z pozostałych danych binarnych
        self.ser.reset_input_buffer()
        
        self.ser.write(b'X')
        
        # Odczytaj odpowiedź, ignorując błędy dekodowania
        try:
            response = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if "STOP" in response:
                print("✓ Pomiary zatrzymane")
            else:
                print(f"✓ Pomiary zatrzymane (odpowiedź: {response})")
        except:
            print("✓ Pomiary zatrzymane")
    
    def status(self):
        """Sprawdź status"""
        self.ser.reset_input_buffer()
        self.ser.write(b'?')
        time.sleep(0.05)
        response = self.ser.readline().decode('utf-8', errors='ignore').strip()
        print(response)
        return response
    
    def reset_mpu(self):
        """Zresetuj i zainicjalizuj ponownie MPU6050"""
        self.ser.reset_input_buffer()
        self.ser.write(b'R')
        time.sleep(0.2)
        response = self.ser.readline().decode('utf-8', errors='ignore').strip()
        if "RESET" in response:
            print("✓ MPU6050 zresetowany")
        else:
            print(f"✓ Reset wysłany (odpowiedź: {response})")
    
    def read_sample(self):
        """
        Odczytaj pojedynczą próbkę
        
        Returns:
            dict: {'timestamp': us, 'ax': g, 'ay': g, 'az': g, 
                   'gx': °/s, 'gy': °/s, 'gz': °/s, 'temp': °C}
        """
        # Szukaj markera 'D'
        marker_found = False
        timeout = 200  # Max prób
        attempts = 0
        
        while not marker_found and attempts < timeout:
            byte = self.ser.read(1)
            if byte == b'D':
                marker_found = True
            elif byte == b'':
                break  # Timeout
            attempts += 1
        
        if not marker_found:
            raise ValueError("Nie znaleziono markera danych")
        
        # Odczytaj timestamp (4 bajty)
        timestamp_bytes = self.ser.read(4)
        if len(timestamp_bytes) != 4:
            raise ValueError("Niekompletny timestamp")
            
        timestamp = struct.unpack('<I', timestamp_bytes)[0]  # unsigned int, little-endian
        
        # Odczytaj 14 bajtów danych
        data_bytes = self.ser.read(14)
        if len(data_bytes) != 14:
            raise ValueError("Niekompletne dane")
        
        # Rozpakuj dane (7 x int16, big-endian)
        raw_data = struct.unpack('>7h', data_bytes)
        
        # Konwertuj na jednostki fizyczne
        sample = {
            'timestamp': timestamp / 1e6,  # mikrosek → sekundy
            'ax': raw_data[0] / self.accel_scale,
            'ay': raw_data[1] / self.accel_scale,
            'az': raw_data[2] / self.accel_scale,
            'temp': raw_data[3] / 340.0 + 36.53,  # Formuła z datasheet
            'gx': raw_data[4] / self.gyro_scale,
            'gy': raw_data[5] / self.gyro_scale,
            'gz': raw_data[6] / self.gyro_scale,
        }
        
        return sample
    
    def collect_samples(self, duration=None, count=None, verbose=True):
        """
        Zbieraj próbki przez określony czas lub liczbę
        
        Args:
            duration: Czas w sekundach (None = nieskończoność)
            count: Liczba próbek (None = nieskończoność)
            verbose: Czy wyświetlać postęp
        
        Returns:
            list: Lista słowników z danymi
        """
        start_time = time.time()
        samples_collected = 0
        last_print = 0
        
        try:
            while self.running:
                # Odczytuj wszystkie dostępne próbki
                while self.ser.in_waiting >= 19:  # 1 + 4 + 14 = 19 bajtów
                    try:
                        sample = self.read_sample()
                        self.data.append(sample)
                        samples_collected += 1
                        
                        # Sprawdź warunki zakończenia
                        if duration and (time.time() - start_time) >= duration:
                            self.running = False
                            break
                        if count and samples_collected >= count:
                            self.running = False
                            break
                    except:
                        # Pomiń uszkodzone próbki
                        continue
                
                # Wyświetl postęp co sekundę
                if verbose and (time.time() - last_print) >= 1.0:
                    elapsed = time.time() - start_time
                    rate = samples_collected / elapsed if elapsed > 0 else 0
                    print(f"  Próbek: {samples_collected}, częstotliwość: {rate:.1f} Hz")
                    last_print = time.time()
                
                # Krótka pauza jeśli brak danych
                if self.ser.in_waiting == 0:
                    time.sleep(0.001)
                    
        except KeyboardInterrupt:
            print("\n⚠ Przerwano przez użytkownika")
        
        print(f"✓ Zebrano {samples_collected} próbek")
        return self.data
    
    def get_sampling_rate(self):
        """Oblicz rzeczywistą częstotliwość próbkowania"""
        if len(self.data) < 2:
            return 0
        
        timestamps = [s['timestamp'] for s in self.data]
        dt = np.diff(timestamps)
        avg_dt = np.mean(dt)
        
        return 1.0 / avg_dt if avg_dt > 0 else 0
    
    def diagnose_data(self):
        """Diagnostyka zebranych danych"""
        if len(self.data) < 2:
            print("✗ Za mało danych do diagnostyki")
            return
        
        timestamps = np.array([s['timestamp'] for s in self.data])
        dt = np.diff(timestamps)
        
        print("\n=== DIAGNOSTYKA DANYCH ===")
        print(f"Liczba próbek: {len(self.data)}")
        print(f"Czas trwania: {timestamps[-1]:.3f} s")
        print(f"Średnia częstotliwość: {self.get_sampling_rate():.1f} Hz")
        print(f"\nCzas między próbkami (dt):")
        print(f"  Średni: {np.mean(dt)*1000:.3f} ms")
        print(f"  Min: {np.min(dt)*1000:.3f} ms")
        print(f"  Max: {np.max(dt)*1000:.3f} ms")
        print(f"  Std: {np.std(dt)*1000:.3f} ms")
        
        # Sprawdź duplikaty
        unique_ts = len(np.unique(timestamps))
        if unique_ts < len(timestamps):
            duplicates = len(timestamps) - unique_ts
            print(f"\n⚠ Duplikaty timestampów: {duplicates} ({duplicates/len(timestamps)*100:.1f}%)")
        else:
            print(f"\n✓ Brak duplikatów timestampów")
        
        # Histogram odstępów czasu
        if len(dt) > 10:
            print(f"\nRozkład dt (ms):")
            hist, bins = np.histogram(dt * 1000, bins=10)
            for i in range(len(hist)):
                bar = '█' * int(hist[i] / max(hist) * 30)
                print(f"  {bins[i]:6.2f}-{bins[i+1]:6.2f}: {bar} ({hist[i]})")
        
        print("========================\n")
    
    def save_to_csv(self, filename='mpu_data.csv'):
        """Zapisz dane do CSV"""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)
        
        print(f"✓ Zapisano do {filename}")
    
    def plot_data(self):
        """Wyświetl wykres danych"""
        if not self.data:
            print("✗ Brak danych do wyświetlenia")
            return
        
        if len(self.data) < 2:
            print("✗ Za mało danych do wykresu (minimum 2 próbki)")
            return
        
        timestamps = np.array([s['timestamp'] for s in self.data])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Sprawdź czy timestampy rosną
        if len(np.unique(timestamps)) < len(timestamps) * 0.5:
            print("⚠ Uwaga: Wiele duplikatów timestampów - może być problem z batchowaniem")
        
        # Akcelerometr
        ax1.plot(timestamps, [s['ax'] for s in self.data], label='X', linewidth=0.8)
        ax1.plot(timestamps, [s['ay'] for s in self.data], label='Y', linewidth=0.8)
        ax1.plot(timestamps, [s['az'] for s in self.data], label='Z', linewidth=0.8)
        ax1.set_ylabel('Przyspieszenie [g]')
        ax1.set_title(f'Akcelerometr (fs = {self.get_sampling_rate():.1f} Hz, n = {len(self.data)})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Żyroskop
        ax2.plot(timestamps, [s['gx'] for s in self.data], label='X', linewidth=0.8)
        ax2.plot(timestamps, [s['gy'] for s in self.data], label='Y', linewidth=0.8)
        ax2.plot(timestamps, [s['gz'] for s in self.data], label='Z', linewidth=0.8)
        ax2.set_xlabel('Czas [s]')
        ax2.set_ylabel('Prędkość kątowa [°/s]')
        ax2.set_title('Żyroskop')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """Zamknij połączenie"""
        try:
            if hasattr(self, 'ser') and self.ser and self.ser.is_open:
                self.ser.close()
                del self.ser  # Usuń referencję do obiektu
                time.sleep(2)  # KLUCZOWE dla Windows - musi być 2s!
            print("✓ Połączenie zamknięte")
        except Exception as e:
            print(f"⚠ Błąd przy zamykaniu: {e}")


# Przykładowe użycie
if __name__ == "__main__":
    mpu = MPU6050Controller(port='COM12')
    
    try:
        mpu.status()
        mpu.start_measurement()
        print("Zbieranie danych...")
        mpu.collect_samples(duration=30)  # Zbieraj przez 30 sekund
        mpu.stop_measurement()
        
        print(f"Częstotliwość próbkowania: {mpu.get_sampling_rate():.1f} Hz")
        print(f"Liczba próbek: {len(mpu.data)}")
        
        mpu.diagnose_data()
        mpu.save_to_csv(filename='anomaly_data_raw\\mpu_data20.csv')
        mpu.plot_data()
        
    except KeyboardInterrupt:
        print("\n⚠ Przerwano przez użytkownika")
    finally:
        mpu.close()