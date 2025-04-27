import numpy as np
import torch
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    def __init__(self, data, labels, window_size=250, purpose="train"):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

        assert window_size < self.data.shape[0], "Invalid window size selected. Window size should be less than signal length."
        self.window_size = window_size

        assert purpose in ["train", "validation", "test"], "Invalid mode selected. Available modes: train, validation, test."
        self.purpose = purpose

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]

        if self.purpose == "train":
            start = np.random.randint(0, signal.shape[0] - self.window_size + 1)
            signal = signal[start:start + self.window_size]

            return signal, label

        elif (self.purpose == "validation") or (self.purpose == "test"):
            step = self.window_size // 2
            windows = signal.unfold(0, self.window_size, step)
            windows = windows.permute(0, 2, 1)

            return windows, label  


class SpectralDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
    