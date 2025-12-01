# data_loader.py
import os
import numpy as np
from sklearn.model_selection import train_test_split


def load_npy_dir(dir_path):
    """
    Wczytuje wszystkie pliki .npy z katalogu i konkatenaje je po osi 0.

    Zakładany format każdego pliku: (n_sequences, window_size, n_features)

    Zwraca:
        ndarray o shape (sum_n_sequences, window_size, n_features)
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    arrays = []
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".npy")])

    if not files:
        raise ValueError(f"No .npy files found in {dir_path}")

    for f in files:
        full_path = os.path.join(dir_path, f)
        arr = np.load(full_path)

        if arr.ndim != 3:
            raise ValueError(
                f"File {f} has wrong shape {arr.shape}. Expected (N, W, F)."
            )

        arrays.append(arr)

    out = np.concatenate(arrays, axis=0)

    # sanity check
    if not np.isfinite(out).all():
        raise ValueError(f"Loaded data from {dir_path} contains NaN/inf.")

    return out


def prepare_datasets(
    train_dir=None,
    anomaly_dir=None,
    val_frac=0.2,
    random_state=42
):
    """
    Ładuje dane normalne oraz anomalie.
    """
    # Domyślne ścieżki
    BASE_DIR = "/app/data"

    train_dir = os.path.join(BASE_DIR, "training_data_refined")
    anomaly_dir = os.path.join(BASE_DIR, "anomaly_data_refined")
    
    print(f"Loading normal data from: {train_dir}")
    X = load_npy_dir(train_dir)

    print("Splitting normal data into train/validation...")
    X_train, X_val = train_test_split(
        X, test_size=val_frac, shuffle=True, random_state=random_state
    )

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    print(f"Train size: {X_train.shape}, Val size: {X_val.shape}")

    # anomalie są opcjonalne
    X_anom = None
    if anomaly_dir and os.path.exists(anomaly_dir):
        print(f"Loading anomaly data from: {anomaly_dir}")
        X_anom = load_npy_dir(anomaly_dir).astype(np.float32)
        print(f"Loaded anomalies: {X_anom.shape}")
    else:
        print("⚠ No anomaly directory found — skipping anomaly dataset.")

    return X_train, X_val, X_anom


if __name__ == "__main__":
    X_train, X_val, X_anom = prepare_datasets()

    print("===================================")
    print("Data Loaded:")
    print("X_train:", X_train.shape)
    print("X_val  :", X_val.shape)
    print("X_anom :", None if X_anom is None else X_anom.shape)
