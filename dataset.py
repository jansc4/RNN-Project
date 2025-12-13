import numpy as np
import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, windows):
        windows = np.array(windows, dtype=np.float32)
        windows[~np.isfinite(windows)] = 0.0 # prevents errors, bc for some reason i got errors about values
        self.windows = torch.from_numpy(windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]
