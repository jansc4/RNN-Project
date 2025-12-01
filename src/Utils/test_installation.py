#!/usr/bin/env python
# test_installation.py
"""
Sprawdź czy wszystkie zależności są zainstalowane i działają
"""

def test_imports():
    """Test importów bibliotek"""
    print("\n" + "="*60)
    print("TEST INSTALACJI - Sprawdzanie bibliotek")
    print("="*60 + "\n")
    
    required = {
        'numpy': 'NumPy',
        'pandas': 'Pandas', 
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'tensorflow': 'TensorFlow',
        'serial': 'PySerial'
    }
    
    failed = []
    
    for module, name in required.items():
        try:
            if module == 'sklearn':
                import sklearn
            elif module == 'serial':
                import serial
            else:
                __import__(module)
            print(f"✓ {name:20} OK")
        except ImportError:
            print(f"✗ {name:20} BRAK")
            failed.append(name)
    
    print("\n" + "="*60)
    
    if failed:
        print("\n⚠ Brakujące biblioteki:")
        for lib in failed:
            print(f"  - {lib}")
        print("\nZainstaluj je przez:")
        print("  pip install " + " ".join([k for k, v in required.items() if v in failed]))
        return False
    else:
        print("\n✓ Wszystkie biblioteki zainstalowane!")
        return True


def test_esp32_connection():
    """Test połączenia z ESP32"""
    print("\n" + "="*60)
    print("TEST POŁĄCZENIA Z ESP32")
    print("="*60 + "\n")
    
    try:
        from src.main import find_esp32_port, check_port_available
        
        port = find_esp32_port()
        
        if port is None:
            print("\n⚠ Nie znaleziono ESP32")
            print("Upewnij się że:")
            print("  1. ESP32 jest podłączony przez USB")
            print("  2. Zainstalowane są sterowniki (CP210x lub CH340)")
            print("  3. Port nie jest używany przez inne aplikacje")
            return False
        
        print(f"\n✓ Znaleziono ESP32 na porcie: {port}")
        
        if check_port_available(port):
            print(f"✓ Port {port} jest dostępny")
            return True
        else:
            print(f"✗ Port {port} nie jest dostępny")
            print("Zamknij inne aplikacje używające tego portu")
            return False
            
    except Exception as e:
        print(f"✗ Błąd: {e}")
        return False


def test_directory_structure():
    """Test struktury katalogów"""
    print("\n" + "="*60)
    print("TEST STRUKTURY KATALOGÓW")
    print("="*60 + "\n")
    
    required_files = [
        'mian.py',
        'pipeline_config.py',
        'data_preprocessing.py',
        'data_loader.py',
        'train_lstm_autoencoder.py',
        'compute_threshold.py',
        'detect_anomalies.py',
        'full_pipeline.py'
    ]
    
    missing = []
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✓ {filename:30} OK")
        else:
            print(f"✗ {filename:30} BRAK")
            missing.append(filename)
    
    print()
    
    if missing:
        print("⚠ Brakujące pliki:")
        for f in missing:
            print(f"  - {f}")
        return False
    else:
        print("✓ Wszystkie pliki na miejscu!")
        return True


def test_config():
    """Test konfiguracji"""
    print("\n" + "="*60)
    print("TEST KONFIGURACJI")
    print("="*60 + "\n")
    
    try:
        from pipeline_config import Config
        
        print("Aktualna konfiguracja:")
        print(f"  Port: {Config.PORT}")
        print(f"  Baudrate: {Config.BAUDRATE}")
        print(f"  Window size: {Config.WINDOW_SIZE}")
        print(f"  Batch size: {Config.BATCH_SIZE}")
        print(f"  Epochs: {Config.EPOCHS}")
        print(f"  Threshold multiplier: {Config.THRESHOLD_MULTIPLIER}")
        
        print("\n✓ Konfiguracja wczytana poprawnie")
        
        # Sprawdź czy port istnieje
        if Config.PORT == 'COM12':
            print("\n⚠ Używasz domyślnego portu COM12")
            print("  Zmień Config.PORT w pipeline_config.py na odpowiedni port")
        
        return True
        
    except Exception as e:
        print(f"✗ Błąd konfiguracji: {e}")
        return False


def run_all_tests():
    """Uruchom wszystkie testy"""
    import os
    
    print("\n" + "="*70)
    print(" "*20 + "TEST INSTALACJI PIPELINE")
    print("="*70)
    
    results = {
        "Biblioteki": test_imports(),
        "Pliki": test_directory_structure(),
        "Konfiguracja": test_config(),
        "ESP32": test_esp32_connection()
    }
    
    print("\n" + "="*70)
    print("PODSUMOWANIE")
    print("="*70 + "\n")
    
    for test_name, result in results.items():
        status = "✓ OK" if result else "✗ FAILED"
        print(f"{test_name:20} {status}")
    
    all_ok = all(results.values())
    
    print("\n" + "="*70)
    
    if all_ok:
        print("\n🎉 WSZYSTKO GOTOWE!")
        print("\nMożesz teraz uruchomić pipeline:")
        print("  python full_pipeline.py --full")
    else:
        print("\n⚠ Niektóre testy nie przeszły")
        print("\nNapraw błędy i uruchom ponownie:")
        print("  python test_installation.py")
    
    print("\n" + "="*70 + "\n")
    
    return all_ok


if __name__ == "__main__":
    import os
    
    # Zmień katalog roboczy jeśli potrzeba
    if not os.path.exists('mian.py'):
        print("⚠ Uruchom ten skrypt z katalogu src/")
        print("  cd Quick_mesure_app/src/")
        print("  python test_installation.py")
        exit(1)
    
    success = run_all_tests()
    exit(0 if success else 1)
