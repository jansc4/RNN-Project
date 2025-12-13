import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config

# loads single parquet file
def load_parquet_file(parquet_path, csv_dir, columns):
    p = Path(parquet_path)
    if not p.is_absolute():
        p = csv_dir / parquet_path

    if not p.exists():
        print(f"Warning: file not found: {p}")
        return None

    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print(f"Warning: failed to read parquet {p}: {e}")
        return None

    missing = [col for col in columns if col not in df.columns]
    if missing:
        print(f"Warning: Missing columns {missing} in {p.name}")
        return None

    arr = df[columns].to_numpy(dtype=np.float32)
    arr[~np.isfinite(arr)] = np.nan
    arr = arr[~np.isnan(arr).any(axis=1)]
    if arr.size == 0:
        return None
    return arr

# builds dataset based on a csv file
def load_dataset_multisensor(csv_path, sensor_config):
    df = pd.read_csv(csv_path)
    csv_dir = Path(csv_path).parent
    sensor_data = {s: {'sequences': [], 'file_names': []}
                   for s, cfg in sensor_config.items() if cfg['enabled']}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {Path(csv_path).name}"):
        for sensor, cfg in sensor_config.items():
            if not cfg['enabled']:
                continue
            fname = row.get(cfg['csv_column'], None)
            if pd.isna(fname) or fname is None:
                continue
            try:
                seq = load_parquet_file(fname, csv_dir, cfg['parquet_columns'])
                if seq is not None and seq.shape[0] >= 1:
                    sensor_data[sensor]['sequences'].append(seq)
                    sensor_data[sensor]['file_names'].append(str(fname))
            except Exception as e:
                print(f"Warning: Failed to load {fname}: {e}")
    return sensor_data

# creates sliding windows
def create_windows(sequences, window_size=config.WINDOW_SIZE, step_size=config.STEP_SIZE):
    all_windows, window_to_file = [], []

    for file_idx, seq in enumerate(sequences):
        if seq is None or len(seq) == 0:
            continue
        seq = np.array(seq, dtype=np.float32)
        seq[~np.isfinite(seq)] = np.nan
        seq = seq[~np.isnan(seq).any(axis=1)]
        if seq.shape[0] < window_size:
            continue
        seq -= seq.mean(axis=0, keepdims=True)
        seq[~np.isfinite(seq)] = 0.0
        last_start = max(0, len(seq) - window_size)
        for start in range(0, last_start + 1, step_size):
            window = seq[start:start + window_size]
            if window.shape[0] == window_size:
                all_windows.append(window.astype(np.float32))
                window_to_file.append(file_idx)

    if len(all_windows) == 0:
        return None, None
    return np.array(all_windows, dtype=np.float32), np.array(window_to_file, dtype=np.int32)
