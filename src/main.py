import argparse
import json
import os
from pathlib import Path
import numpy as np

from data_loader import prepare_datasets
from prepare_sequences import prepare_sequences
from train_lstm_autoencoder import train_model
from compute_threshold import compute_threshold
from detect_anomalies import detect_from_file


# Ścieżki względem głównego katalogu /app
BASE_DIR = Path(__file__).parent.parent
OUTPUT_MODELS = BASE_DIR / "outputs" / "models"
OUTPUT_PLOTS = BASE_DIR / "outputs" / "plots"
OUTPUT_LOGS = BASE_DIR / "outputs" / "logs"
THRESHOLD_FILE = OUTPUT_MODELS / "threshold.json"


def ensure_dirs():
    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOGS.mkdir(parents=True, exist_ok=True)




def run_prepare():
    print("[1/4] Preparing data...")

    X_train, X_val, X_anom = prepare_datasets()

    # skalujemy tylko na TRAIN
    X_train_s, scaler = prepare_sequences(X_train)

    # stosujemy ten sam scaler na VALIDATION
    X_val_s = scaler.transform(X_val.reshape(-1, X_val.shape[2])).reshape(X_val.shape)

    if X_anom is not None:
        X_anom_s = scaler.transform(X_anom.reshape(-1, X_anom.shape[2])).reshape(X_anom.shape)

    np.save(OUTPUT_MODELS / "train_sequences.npy", X_train_s)
    np.save(OUTPUT_MODELS / "val_sequences.npy", X_val_s)
    if X_anom is not None:
        np.save(OUTPUT_MODELS / "anom_sequences.npy", X_anom_s)

    import pickle
    with open(OUTPUT_MODELS / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("[OK] Zapisano sekwencje i scaler.")



def run_train():
    print("[2/4] Training LSTM autoencoder...")

    import numpy as np
    seq_train = np.load(OUTPUT_MODELS / "train_sequences.npy")
    seq_val   = np.load(OUTPUT_MODELS / "val_sequences.npy")

    model, history = train_model(seq_train, seq_val, MODEL_PATH=str(OUTPUT_MODELS / "lstm_autoencoder.h5"))

    model_path = OUTPUT_MODELS / "lstm_autoencoder.h5"
    model.save(model_path)
    print(f"Model saved to {model_path}")


def run_threshold():
    print("[3/4] Computing threshold...")
    import numpy as np

    seq_path = OUTPUT_MODELS / "train_sequences.npy"
    sequences = np.load(seq_path)

    threshold = compute_threshold(sequences, MODEL_PATH=str(OUTPUT_MODELS / "lstm_autoencoder.h5"))

    with open(THRESHOLD_FILE, "w") as f:
        json.dump({"threshold": threshold}, f)

    print(f"Threshold saved to {THRESHOLD_FILE}")


def run_detect():
    print("[4/4] Detecting anomalies...")
    import numpy as np
    anomaly_df = np.load(OUTPUT_MODELS / "anom_sequences.npy")

    # Load scaler
    import pickle
    with open(OUTPUT_MODELS / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load threshold
    with open(THRESHOLD_FILE, "r") as f:
        threshold = json.load(f)["threshold"]

    results = detect_from_file(anomaly_df, scaler, threshold, MODEL_PATH=str(OUTPUT_MODELS / "lstm_autoencoder.h5"))

    out_file = OUTPUT_PLOTS / "anomaly_detection_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="LSTM anomaly detection pipeline")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--threshold", action="store_true")
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    ensure_dirs()

    if args.all or args.prepare:
        run_prepare()

    if args.all or args.train:
        run_train()

    if args.all or args.threshold:
        run_threshold()

    if args.all or args.detect:
        run_detect()


if __name__ == "__main__":
    main()
