import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import config


def train_model(model, train_loader, val_loader, device, sensor_name, epochs=config.EPOCHS, patience=config.PATIENCE):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-6)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    if config.ENABLE_AUTOGRAD_DETECT_ANOMALY:
        torch.autograd.set_detect_anomaly(True)

    print(f"Training {sensor_name.upper()} model")

    for epoch in range(epochs): # epoch processing
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")
        for batch in train_pbar:    # batch processing
            batch = batch.to(device)
            optimizer.zero_grad()
            output, _ = model(batch)
            batch = torch.clamp(batch, -1e6, 1e6)   # clamp extreme values
            output = torch.clamp(output, -1e6, 1e6)

            loss = criterion(output, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("NaN/Inf loss detected.")

            loss.backward() # backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        train_loss /= len(train_loader) if len(train_loader) > 0 else 1
        train_losses.append(train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", unit="batch")
        with torch.no_grad():
            for batch in val_pbar:
                batch = batch.to(device)
                output, _ = model(batch)
                batch = torch.clamp(batch, -1e6, 1e6)
                output = torch.clamp(output, -1e6, 1e6)
                loss = criterion(output, batch)
                if not torch.isfinite(loss):
                    loss = torch.tensor(1e6)
                val_loss += float(loss.item())
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"  New best model (val_loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    if config.ENABLE_AUTOGRAD_DETECT_ANOMALY:
        torch.autograd.set_detect_anomaly(False)

    return model, train_losses, val_losses

# how model reconstruct each input
def compute_reconstruction_errors(model, loader, device, sensor_name):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Computing {sensor_name} errors"):
            batch = batch.to(device)
            output, _ = model(batch)
            batch = torch.clamp(batch, -1e6, 1e6)
            output = torch.clamp(output, -1e6, 1e6)
            mse = torch.mean((output - batch) ** 2, dim=(1,2))  # mse per sample
            arr = mse.cpu().numpy()
            arr[~np.isfinite(arr)] = np.nan
            errors.extend(arr.tolist())
    return np.array(errors, dtype=np.float64)

# aggregates window-level errors to file-level scores
def aggregate_file_scores(window_errors, window_to_file):
    file_scores = []
    if window_errors is None or window_to_file is None:
        return np.array([])
    for file_idx in np.unique(window_to_file):
        file_windows = window_errors[window_to_file==file_idx]
        file_windows = file_windows[np.isfinite(file_windows)]
        file_scores.append(np.nan if file_windows.size==0 else max(np.mean(file_windows), np.percentile(file_windows, 95))) # one score per file
    return np.array(file_scores, dtype=np.float64)
