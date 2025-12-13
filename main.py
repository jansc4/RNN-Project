from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torch

import config
from dataset import WindowDataset
from loader import load_dataset_multisensor, create_windows
from lstm_autoencoder import LSTMAutoencoder
from torch_utils import init_weights
from train import train_model, compute_reconstruction_errors, aggregate_file_scores
from viz import (
    plot_training_history,
    plot_error_distributions,
    plot_roc_curve,
    plot_confusion_matrix
)

def main():
    cfg = config
    device = cfg.DEVICE

    base_dir = Path(__file__).resolve().parent  # same directory as code

    # sensors
    enabled_sensors = [name for name, c in cfg.SENSOR_CONFIG.items() if c['enabled']]
    if not enabled_sensors:
        print("No sensors enabled. Exiting.")
        return

    print(f"\nEnabled sensors: {', '.join(enabled_sensors)}")

    train_data = load_dataset_multisensor(cfg.TRAIN_CSV, cfg.SENSOR_CONFIG)
    normal_data = load_dataset_multisensor(cfg.TEST_NORMAL_CSV, cfg.SENSOR_CONFIG)
    anomaly_data = load_dataset_multisensor(cfg.TEST_ANOMALY_CSV, cfg.SENSOR_CONFIG)

    models, scalers, thresholds = {}, {}, {}
    histories = []
    sensor_normal_file_scores, sensor_anomaly_file_scores = {}, {}


    for sensor in enabled_sensors:
        print(f"\n{'='*70}\nPROCESSING {sensor.upper()}\n{'='*70}")

        X_train, train_w2f = create_windows(train_data[sensor]['sequences'])
        X_normal, normal_w2f = create_windows(normal_data[sensor]['sequences'])
        X_anomaly, anomaly_w2f = create_windows(anomaly_data[sensor]['sequences'])

        if X_train is None or X_normal is None or X_anomaly is None:
            print(f"Skipping {sensor}: insufficient windows.")
            continue

        # normalization
        scaler = StandardScaler()
        n_channels = X_train.shape[2]

        flat_train = np.nan_to_num(
            X_train.reshape(-1, n_channels),
            nan=0.0, posinf=1e6, neginf=-1e6
        )
        scaler.fit(flat_train)
        scaler.scale_[scaler.scale_ == 0] = 1.0

        def safe_transform(arr):
            flat = np.nan_to_num(arr.reshape(-1, n_channels), nan=0.0)
            out = scaler.transform(flat)
            out[~np.isfinite(out)] = 0.0
            return out.reshape(arr.shape)

        X_train, X_normal, X_anomaly = map(
            safe_transform, [X_train, X_normal, X_anomaly]
        )

        scalers[sensor] = scaler

        # dataloaders
        pin_mem = device.type == 'cuda'
        train_loader = DataLoader(WindowDataset(X_train), cfg.BATCH_SIZE, True, pin_memory=pin_mem)
        val_loader   = DataLoader(WindowDataset(X_normal), cfg.BATCH_SIZE, False, pin_memory=pin_mem)
        test_loader  = DataLoader(WindowDataset(X_anomaly), cfg.BATCH_SIZE, False, pin_memory=pin_mem)

        # model
        model = LSTMAutoencoder(input_dim=n_channels).to(device)
        model.apply(init_weights)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        try:
            model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, device, sensor
            )
        except Exception as e:
            print(f"Training failed for {sensor}: {e}")
            continue

        # save model
        model_path = base_dir / f"model_{sensor}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": n_channels,
            "sensor": sensor
        }, model_path)

        print(f" Model saved: {model_path.name}")

        models[sensor] = model
        histories.append((train_losses, val_losses))

        # evaluation
        normal_errors  = compute_reconstruction_errors(model, val_loader, device, sensor)
        anomaly_errors = compute_reconstruction_errors(model, test_loader, device, sensor)

        finite = normal_errors[np.isfinite(normal_errors)]
        threshold = np.mean(finite) + 2 * np.std(finite) if finite.size else np.inf # anomaly threshold
        thresholds[sensor] = threshold

        sensor_normal_file_scores[sensor]  = aggregate_file_scores(normal_errors, normal_w2f)
        sensor_anomaly_file_scores[sensor] = aggregate_file_scores(anomaly_errors, anomaly_w2f)

        # per-sensor metrics
        y_true = np.r_[np.zeros(len(sensor_normal_file_scores[sensor])),
                       np.ones(len(sensor_anomaly_file_scores[sensor]))]
        y_scores = np.r_[sensor_normal_file_scores[sensor],
                         sensor_anomaly_file_scores[sensor]]
        y_scores[~np.isfinite(y_scores)] = np.inf
        y_pred = (y_scores > threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        try:
            roc_auc = roc_auc_score(y_true[np.isfinite(y_scores)],
                                    y_scores[np.isfinite(y_scores)])
        except:
            roc_auc = float('nan')

        print(f"\n{sensor.capitalize()} Performance:")
        print(f"  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  ROC-AUC={roc_auc}")

    # late fusion
    if not sensor_normal_file_scores:
        print("No sensors successfully processed.")
        return

    enabled_sensors = list(sensor_normal_file_scores.keys())
    n_normal = min(len(sensor_normal_file_scores[s]) for s in enabled_sensors)
    n_anom   = min(len(sensor_anomaly_file_scores[s]) for s in enabled_sensors)

    normal_flags  = np.zeros((n_normal, len(enabled_sensors)), bool)
    anomaly_flags = np.zeros((n_anom,   len(enabled_sensors)), bool)

    for i, s in enumerate(enabled_sensors):
        normal_flags[:, i]  = sensor_normal_file_scores[s][:n_normal]  > thresholds[s]
        anomaly_flags[:, i] = sensor_anomaly_file_scores[s][:n_anom]   > thresholds[s]

    y_true = np.r_[np.zeros(n_normal), np.ones(n_anom)]
    y_pred_any = np.r_[np.any(normal_flags, 1), np.any(anomaly_flags, 1)].astype(int)

    plot_confusion_matrix(y_true, y_pred_any, "Late Fusion – ANY Sensor")

    if histories:
        plot_training_history(histories, enabled_sensors)


if __name__ == "__main__":
    main()
