# src/datasets.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample


def load_and_prepare_data(file_path: str,
                          time_steps: int = 50,
                          seed: int = 42):
    """
    Load sliced_data.csv, normalize features, build MANY-to-ONE sequences,
    balance classes via downsampling, and split into train/test sets.
    """
    df = pd.read_csv(file_path) #'sliced_data.csv'

    X = df[
        [
            'L ankle JC', 'L ankle JC.1', 'L ankle JC.2',
            'L elbow JC', 'L elbow JC.1', 'L elbow JC.2',
            'L knee JC', 'L knee JC.1', 'L knee JC.2',
            'R ankle JC', 'R ankle JC.1', 'R ankle JC.2',
            'R elbow JC', 'R elbow JC.1', 'R elbow JC.2',
            'R knee JC', 'R knee JC.1', 'R knee JC.2',
        ]
    ].values
    y = df['binary'].values

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # MANY-to-ONE sequences
    X_sequences = []
    y_sequences = []
    for i in range(len(X_scaled) - time_steps + 1):
        X_sequences.append(X_scaled[i: i + time_steps])
        y_sequences.append(y[i + time_steps - 1])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    # class balancing via downsampling
    class_0_idx = np.where(y_sequences == 0)[0]
    class_1_idx = np.where(y_sequences == 1)[0]

    class_0_down = resample(
        class_0_idx,
        replace=False,
        n_samples=len(class_1_idx),
        random_state=seed,
    )

    resampled_idx = np.hstack([class_0_down, class_1_idx])
    X_resampled = X_sequences[resampled_idx]
    y_resampled = y_sequences[resampled_idx]

    # train/test split
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(X_resampled, y_resampled))

    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]

    return X_train, X_test, y_train, y_test


def get_dataloaders(X_train, X_test, y_train, y_test,
                    batch_size: int = 256):
    """
    Convert numpy arrays to torch tensors and wrap them in DataLoaders.
    """
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader
