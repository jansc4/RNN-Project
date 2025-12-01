# visualize_results.py
"""
Wizualizacja wyników treningu i detekcji
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import prepare_datasets
from compute_threshold import recon_errors
from pipeline_config import Config
import os

def plot_training_history():
    """Wyświetl historię treningu"""
    if not os.path.exists('training_history.json'):
        print("✗ Brak pliku training_history.json")
        return
    
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Historia Treningu LSTM Autoencoder')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("✓ Zapisano: training_history.png")
    plt.show()


def plot_reconstruction_errors():
    """Wyświetl rozkład błędów rekonstrukcji"""
    if not os.path.exists(Config.MODEL_PATH):
        print(f"✗ Brak modelu: {Config.MODEL_PATH}")
        return
    
    print("Wczytywanie danych...")
    model = tf.keras.models.load_model(Config.MODEL_PATH)
    X_train, X_val, X_anom = prepare_datasets(val_frac=0.15)
    
    print("Obliczanie błędów...")
    val_err = recon_errors(model, X_val)
    
    # Wczytaj próg
    if os.path.exists('threshold.json'):
        with open('threshold.json', 'r') as f:
            threshold = json.load(f)['threshold']
    else:
        threshold = val_err.mean() + 3 * val_err.std()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram błędów walidacyjnych
    ax = axes[0, 0]
    ax.hist(val_err, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Próg: {threshold:.6f}')
    ax.set_xlabel('MSE')
    ax.set_ylabel('Liczba próbek')
    ax.set_title('Rozkład Błędów - Dane Walidacyjne (normalne)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Błędy anomalii (jeśli dostępne)
    ax = axes[0, 1]
    if X_anom is not None:
        anom_err = recon_errors(model, X_anom)
        ax.hist(anom_err, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Próg: {threshold:.6f}')
        ax.set_xlabel('MSE')
        ax.set_ylabel('Liczba próbek')
        ax.set_title('Rozkład Błędów - Dane Anomalii')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Brak danych anomalii', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Dane Anomalii')
    
    # 3. Porównanie rozkładów
    ax = axes[1, 0]
    ax.hist(val_err, bins=50, alpha=0.5, color='blue', label='Normalne', edgecolor='black')
    if X_anom is not None:
        ax.hist(anom_err, bins=50, alpha=0.5, color='red', label='Anomalie', edgecolor='black')
    ax.axvline(threshold, color='green', linestyle='--', linewidth=2, label='Próg')
    ax.set_xlabel('MSE')
    ax.set_ylabel('Liczba próbek')
    ax.set_title('Porównanie Rozkładów')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 4. Statystyki
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
STATYSTYKI BŁĘDÓW

Dane Walidacyjne (normalne):
  Średnia: {val_err.mean():.6f}
  Mediana: {np.median(val_err):.6f}
  Std: {val_err.std():.6f}
  Min: {val_err.min():.6f}
  Max: {val_err.max():.6f}
  
Próg detekcji: {threshold:.6f}
False Positive Rate: {np.sum(val_err > threshold)/len(val_err)*100:.2f}%
"""
    
    if X_anom is not None:
        detection_rate = np.sum(anom_err > threshold) / len(anom_err) * 100
        stats_text += f"""
Dane Anomalii:
  Średnia: {anom_err.mean():.6f}
  Mediana: {np.median(anom_err):.6f}
  Std: {anom_err.std():.6f}
  Min: {anom_err.min():.6f}
  Max: {anom_err.max():.6f}
  
Detection Rate: {detection_rate:.1f}%
"""
    
    ax.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top', 
            family='monospace', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('reconstruction_errors.png', dpi=150)
    print("✓ Zapisano: reconstruction_errors.png")
    plt.show()


def plot_example_reconstructions(n_examples=3):
    """Wyświetl przykłady rekonstrukcji"""
    if not os.path.exists(Config.MODEL_PATH):
        print(f"✗ Brak modelu: {Config.MODEL_PATH}")
        return
    
    print("Wczytywanie danych...")
    model = tf.keras.models.load_model(Config.MODEL_PATH)
    X_train, X_val, X_anom = prepare_datasets(val_frac=0.15)
    
    # Wybierz losowe przykłady
    idx_normal = np.random.choice(len(X_val), n_examples, replace=False)
    
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 4*n_examples))
    
    feature_names = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    
    for i, idx in enumerate(idx_normal):
        original = X_val[idx]
        reconstructed = model.predict(X_val[idx:idx+1], verbose=0)[0]
        mse = np.mean(np.square(original - reconstructed))
        
        # Akcelerometr
        ax = axes[i, 0] if n_examples > 1 else axes[0]
        for j in range(3):
            ax.plot(original[:, j], color=colors[j], linestyle='-', alpha=0.7, label=f'{feature_names[j]} (orig)')
            ax.plot(reconstructed[:, j], color=colors[j], linestyle='--', alpha=0.7, label=f'{feature_names[j]} (rec)')
        ax.set_ylabel('Przyspieszenie [g]')
        ax.set_title(f'Przykład {i+1}: Akcelerometr (MSE: {mse:.6f})')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Żyroskop
        ax = axes[i, 1] if n_examples > 1 else axes[1]
        for j in range(3, 6):
            ax.plot(original[:, j], color=colors[j], linestyle='-', alpha=0.7, label=f'{feature_names[j]} (orig)')
            ax.plot(reconstructed[:, j], color=colors[j], linestyle='--', alpha=0.7, label=f'{feature_names[j]} (rec)')
        ax.set_ylabel('Prędkość kątowa [°/s]')
        ax.set_title(f'Przykład {i+1}: Żyroskop')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Błąd per feature
        ax = axes[i, 2] if n_examples > 1 else axes[2]
        error_per_feature = np.mean(np.square(original - reconstructed), axis=0)
        bars = ax.bar(feature_names, error_per_feature, color=colors)
        ax.set_ylabel('MSE')
        ax.set_title(f'Przykład {i+1}: Błąd na cechę')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('example_reconstructions.png', dpi=150)
    print("✓ Zapisano: example_reconstructions.png")
    plt.show()


def plot_anomaly_examples(n_examples=3):
    """Wyświetl przykłady anomalii"""
    if not os.path.exists(Config.MODEL_PATH):
        print(f"✗ Brak modelu: {Config.MODEL_PATH}")
        return
    
    print("Wczytywanie danych...")
    model = tf.keras.models.load_model(Config.MODEL_PATH)
    X_train, X_val, X_anom = prepare_datasets(val_frac=0.15)
    
    if X_anom is None:
        print("✗ Brak danych anomalii")
        return
    
    # Oblicz błędy i wybierz największe
    anom_err = recon_errors(model, X_anom)
    idx_worst = np.argsort(anom_err)[-n_examples:]
    
    # Wczytaj próg
    if os.path.exists('threshold.json'):
        with open('threshold.json', 'r') as f:
            threshold = json.load(f)['threshold']
    else:
        val_err = recon_errors(model, X_val)
        threshold = val_err.mean() + 3 * val_err.std()
    
    fig, axes = plt.subplots(n_examples, 2, figsize=(14, 4*n_examples))
    
    feature_names = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    
    for i, idx in enumerate(idx_worst):
        original = X_anom[idx]
        reconstructed = model.predict(X_anom[idx:idx+1], verbose=0)[0]
        mse = anom_err[idx]
        
        # Wszystkie cechy
        ax = axes[i, 0] if n_examples > 1 else axes[0]
        for j, (name, color) in enumerate(zip(feature_names, colors)):
            ax.plot(original[:, j], color=color, linestyle='-', alpha=0.7, linewidth=1.5, label=f'{name}')
        ax.set_ylabel('Wartość znormalizowana')
        ax.set_title(f'Anomalia {i+1}: Oryginał (MSE: {mse:.6f}, próg: {threshold:.6f})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Błąd rekonstrukcji
        ax = axes[i, 1] if n_examples > 1 else axes[1]
        error = np.abs(original - reconstructed)
        for j, (name, color) in enumerate(zip(feature_names, colors)):
            ax.plot(error[:, j], color=color, alpha=0.7, linewidth=1.5, label=f'{name}')
        ax.set_ylabel('Błąd bezwzględny')
        ax.set_title(f'Anomalia {i+1}: Błąd rekonstrukcji')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_examples.png', dpi=150)
    print("✓ Zapisano: anomaly_examples.png")
    plt.show()


def main():
    """Menu wizualizacji"""
    print("\n" + "="*60)
    print("WIZUALIZACJA WYNIKÓW")
    print("="*60)
    print("\n1. Historia treningu")
    print("2. Rozkład błędów rekonstrukcji")
    print("3. Przykłady rekonstrukcji (normalne)")
    print("4. Przykłady anomalii")
    print("5. Wszystkie wykresy")
    print("0. Wyjście")
    
    choice = input("\nWybierz opcję: ").strip()
    
    if choice == '1':
        plot_training_history()
    elif choice == '2':
        plot_reconstruction_errors()
    elif choice == '3':
        plot_example_reconstructions()
    elif choice == '4':
        plot_anomaly_examples()
    elif choice == '5':
        plot_training_history()
        plot_reconstruction_errors()
        plot_example_reconstructions()
        plot_anomaly_examples()
    else:
        print("Wyjście")


if __name__ == "__main__":
    main()
