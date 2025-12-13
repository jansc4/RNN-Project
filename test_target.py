import torch
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

import config
from dataset import WindowDataset
from loader import load_dataset_multisensor, create_windows
from lstm_autoencoder import LSTMAutoencoder
from torch_utils import init_weights
from train import compute_reconstruction_errors, aggregate_file_scores
from viz import plot_roc_curve, plot_confusion_matrix
import matplotlib.pyplot as plt


def plot_fusion_error_distributions(sensor_normal_file_scores, sensor_anomaly_file_scores, thresholds):
    sensors = list(sensor_normal_file_scores.keys())
    n_normal = min(len(sensor_normal_file_scores[s]) for s in sensors)
    n_anom = min(len(sensor_anomaly_file_scores[s]) for s in sensors)

    normal_scores = np.vstack([sensor_normal_file_scores[s][:n_normal] for s in sensors])
    anomaly_scores = np.vstack([sensor_anomaly_file_scores[s][:n_anom] for s in sensors])
    fused_normal = np.nanmax(normal_scores, axis=0)
    fused_anomaly = np.nanmax(anomaly_scores, axis=0)

    global_thresh = max(thresholds.values())

    plt.figure(figsize=(8,4))
    plt.hist(fused_normal, bins=50, alpha=0.6, label='Normal', density=True)
    plt.hist(fused_anomaly, bins=50, alpha=0.6, label='Anomaly', density=True)
    if np.isfinite(global_thresh):
        plt.axvline(global_thresh, color='r', linestyle='--', label='Threshold')
    plt.title("Late Fusion Reconstruction Error - Linear Scale")
    plt.xlabel("Fused File Score (MSE)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(8,4))
    plt.hist(fused_normal, bins=50, alpha=0.6, label='Normal', density=True)
    plt.hist(fused_anomaly, bins=50, alpha=0.6, label='Anomaly', density=True)
    if np.isfinite(global_thresh):
        plt.axvline(global_thresh, color='r', linestyle='--', label='Threshold')
    plt.yscale('log')
    plt.title("Late Fusion Reconstruction Error - Log Scale")
    plt.xlabel("Fused File Score (MSE)")
    plt.ylabel("Density (log)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def main():
    cfg = config
    device = cfg.DEVICE
    base_dir = Path(__file__).resolve().parent

    enabled_sensors = [name for name, c in cfg.SENSOR_CONFIG.items() if c['enabled']]
    if not enabled_sensors:
        print("No sensors enabled. Exiting.")
        return
    print(f"Enabled sensors: {', '.join(enabled_sensors)}")

    normal_data = load_dataset_multisensor(cfg.TEST_NORMAL_CSV, cfg.SENSOR_CONFIG)
    anomaly_data = load_dataset_multisensor(cfg.TEST_ANOMALY_CSV, cfg.SENSOR_CONFIG)

    models, scalers, thresholds = {}, {}, {}
    sensor_normal_file_scores, sensor_anomaly_file_scores = {}, {}

    for sensor in enabled_sensors:
        model_path = base_dir / f"model_{sensor}.pth"
        if not model_path.exists():
            print(f"Model for {sensor} not found: {model_path}")
            continue

        checkpoint = torch.load(model_path, map_location=device)
        n_channels = checkpoint["input_dim"]
        model = LSTMAutoencoder(input_dim=n_channels).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models[sensor] = model

        # create windows
        X_normal, normal_w2f = create_windows(normal_data[sensor]['sequences'])
        X_anomaly, anomaly_w2f = create_windows(anomaly_data[sensor]['sequences'])
        if X_normal is None or X_anomaly is None:
            print(f"Skipping {sensor}: insufficient windows.")
            continue

        # normalize using normal data
        scaler = StandardScaler()
        flat_data = np.nan_to_num(X_normal.reshape(-1, n_channels), nan=0.0, posinf=1e6, neginf=-1e6)
        scaler.fit(flat_data)
        scaler.scale_[scaler.scale_ == 0] = 1.0
        scalers[sensor] = scaler

        def safe_transform(arr):
            flat = np.nan_to_num(arr.reshape(-1, n_channels), nan=0.0)
            out = scaler.transform(flat)
            out[~np.isfinite(out)] = 0.0
            return out.reshape(arr.shape)

        X_normal, X_anomaly = map(safe_transform, [X_normal, X_anomaly])

        pin_mem = device.type == 'cuda'
        normal_loader = DataLoader(WindowDataset(X_normal), cfg.BATCH_SIZE, False, pin_memory=pin_mem)
        anomaly_loader = DataLoader(WindowDataset(X_anomaly), cfg.BATCH_SIZE, False, pin_memory=pin_mem)

        normal_errors = compute_reconstruction_errors(model, normal_loader, device, sensor)
        anomaly_errors = compute_reconstruction_errors(model, anomaly_loader, device, sensor)

        # threshold for this sensor
        finite = normal_errors[np.isfinite(normal_errors)]
        threshold = np.mean(finite) + 2 * np.std(finite) if finite.size else np.inf
        thresholds[sensor] = threshold

        # aggregate to file-level
        sensor_normal_file_scores[sensor] = aggregate_file_scores(normal_errors, normal_w2f)
        sensor_anomaly_file_scores[sensor] = aggregate_file_scores(anomaly_errors, anomaly_w2f)

    if not sensor_normal_file_scores:
        print("No sensors successfully processed.")
        return

    enabled_sensors = list(sensor_normal_file_scores.keys())
    n_normal = min(len(sensor_normal_file_scores[s]) for s in enabled_sensors)
    n_anom = min(len(sensor_anomaly_file_scores[s]) for s in enabled_sensors)

    normal_flags = np.zeros((n_normal, len(enabled_sensors)), bool)
    anomaly_flags = np.zeros((n_anom, len(enabled_sensors)), bool)

    for i, s in enumerate(enabled_sensors):
        normal_flags[:, i] = sensor_normal_file_scores[s][:n_normal] > thresholds[s]
        anomaly_flags[:, i] = sensor_anomaly_file_scores[s][:n_anom] > thresholds[s]

    # ANY sensor triggers anomaly
    y_true = np.r_[np.zeros(n_normal), np.ones(n_anom)]
    y_pred_any = np.r_[np.any(normal_flags, 1), np.any(anomaly_flags, 1)].astype(int)

    # metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_any, average='binary', zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_pred_any)
    except:
        roc_auc = float('nan')

    print("\nLate Fusion (ANY sensor triggers anomaly) Performance on Target Data:")
    print(f"  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  ROC-AUC={roc_auc:.4f}")

    # visualizations
    plot_confusion_matrix(y_true, y_pred_any, title="Late Fusion – ANY Sensor")
    plot_roc_curve(y_true, y_pred_any, title="Late Fusion ROC Curve")

    # fused reconstruction error distributions
    plot_fusion_error_distributions(sensor_normal_file_scores, sensor_anomaly_file_scores, thresholds)


if __name__ == "__main__":
    main()
