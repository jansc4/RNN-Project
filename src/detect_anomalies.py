# detect_anomalies.py
import numpy as np
import tensorflow as tf
from compute_threshold import recon_errors



def detect_from_file(anomaly_sequences, scaler, threshold, MODEL_PATH):
    """
    Detekcja anomalii na przekazanych sekwencjach.
    
    Args:
        anomaly_sequences: numpy array z sekwencjami do sprawdzenia
        scaler: obiekt skalera (nie jest używany, ale zachowany dla kompatybilności)
        threshold: próg detekcji
    
    Returns:
        dict: wyniki detekcji
    """
    print("\n" + "="*60)
    print("DETEKCJA ANOMALII")
    print("="*60)
    
    # Wczytaj model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✓ Wczytano model: {MODEL_PATH}")
    print(f"✓ Próg: {threshold:.6f}")
    print(f"✓ Sekwencje: {anomaly_sequences.shape}")
    
    # Oblicz błędy
    print("\nObliczanie błędów rekonstrukcji...")
    errors = recon_errors(model, anomaly_sequences)
    
    detected = np.sum(errors > threshold)
    total = len(errors)
    detection_rate = detected / total * 100
    
    print(f"\nWykryto: {detected}/{total} ({detection_rate:.1f}%)")
    print(f"Błąd średni: {errors.mean():.6f}")
    print(f"Błąd min: {errors.min():.6f}")
    print(f"Błąd max: {errors.max():.6f}")
    
    # Przygotuj wyniki
    results = {
        "threshold": float(threshold),
        "total_sequences": int(total),
        "detected": int(detected),
        "detection_rate": float(detection_rate),
        "mean_error": float(errors.mean()),
        "min_error": float(errors.min()),
        "max_error": float(errors.max()),
        "detections": []
    }
    
    for i, (err, is_anom) in enumerate(zip(errors, errors > threshold)):
        results["detections"].append({
            "sequence_id": int(i),
            "error": float(err),
            "is_anomaly": bool(is_anom)
        })
    
    print("\n" + "="*60)
    
    return results


# Zachowanie kompatybilności wstecznej
def detect_anomalies_batch(model=None, X_anom=None, threshold=None):
    """Stara funkcja - dla kompatybilności"""
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    
    if X_anom is None:
        from data_loader import prepare_datasets
        _, X_val, X_anom = prepare_datasets(val_frac=0.15)
    
    if threshold is None:
        print("⚠ Brak progu! Użyj compute_threshold()")
        return None
    
    return detect_from_file(X_anom, None, threshold)


if __name__ == "__main__":
    print("Uruchom poprzez main.py --detect")