# compute_threshold.py
import numpy as np
import tensorflow as tf

THRESHOLD_MULTIPLIER = 3.0

def recon_errors(model, X):
    """Oblicz błędy rekonstrukcji"""
    X_pred = model.predict(X, verbose=0)
    mse = np.mean(np.square(X - X_pred), axis=(1, 2))
    return mse


def compute_threshold(sequences, MODEL_PATH):
    """
    Oblicz próg detekcji anomalii na podstawie przekazanych sekwencji.
    
    Args:
        sequences: numpy array z sekwencjami treningowymi/walidacyjnymi
    
    Returns:
        float: obliczony próg
    """
    print("\n" + "="*60)
    print("OBLICZANIE PROGU DETEKCJI")
    print("="*60)
    
    # Wczytaj model
    print(f"\nWczytywanie modelu: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Model wczytany")
    
    print(f"✓ Sequences shape: {sequences.shape}")
    
    # Oblicz błędy
    print("\nObliczanie błędów rekonstrukcji...")
    errors = recon_errors(model, sequences)
    
    mean = np.mean(errors)
    std = np.std(errors)
    median = np.median(errors)
    
    threshold = mean + THRESHOLD_MULTIPLIER * std
    
    print("\n" + "="*60)
    print("STATYSTYKI BŁĘDÓW")
    print("="*60)
    print(f"Średnia (mean):        {mean:.6f}")
    print(f"Odchylenie (std):      {std:.6f}")
    print(f"Mediana:               {median:.6f}")
    print(f"\nPRÓG (mean + {THRESHOLD_MULTIPLIER}*std): {threshold:.6f}")
    print("="*60)
    
    return threshold


if __name__ == "__main__":
    from data_loader import prepare_datasets
    X_train, X_val, X_anom = prepare_datasets(val_frac=0.15)
    threshold = compute_threshold(X_val)
    print(f"\nObliczony próg: {threshold}")