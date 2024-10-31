import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from config import Config
import polars as pl

class BaseDataset(Dataset):
    def __init__(self, dir, seq_len, columns):
        self.seq_len = seq_len
        data = []
        with open(dir, 'r') as file:
            next(file)  # Skip header
            for line in file:
                try:
                    row = [float(line.split(',')[col]) for col in columns]
                    data.append(row)
                except ValueError:
                    continue  # Skip non-numeric rows
        data = np.array(data)
        self.X = self.normalize(data[:, :-1])
        self.y = data[:, [-1]]
        self.len = len(self.y)

    def __len__(self):
        return self.len - self.seq_len - 1

    def __getitem__(self, idx):
        x = np.transpose(self.X[idx:idx + self.seq_len])
        label = self.y[idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    def normalize(self, X):
        X = np.transpose(X)
        X_norm = [(x - np.min(x)) / (np.max(x) - np.min(x)) for x in X]
        return np.transpose(X_norm)

class Uber(BaseDataset):
    def __init__(self, dir, seq_len=60):
        super().__init__(dir, seq_len, columns=[1, 2, 3, 4, 5])  # Adjust these indices based on your CSV structure

class IstanbulStock(BaseDataset):
    def __init__(self, dir, seq_len=40):
        super().__init__(dir, seq_len, columns=range(8))

class AirQuality(BaseDataset):
    def __init__(self, dir=None, seq_len=24):
        try:
            from ucimlrepo import fetch_ucirepo

            # Fetch Beijing PM2.5 dataset
            beijing_pm2_5 = fetch_ucirepo(id=381)

            # Get features DataFrame and convert to polars
            features_df = pl.from_pandas(beijing_pm2_5.data.features)

            # Convert wind direction to one-hot encoding
            # Get unique wind directions and create dummy columns
            wind_dummies = features_df.get_column('cbwd').unique()
            for direction in wind_dummies:
                features_df = features_df.with_columns(
                    pl.when(pl.col('cbwd') == direction)
                      .then(1)
                      .otherwise(0)
                      .alias(f'wind_{direction}')
                )

            # Drop original categorical column
            features_df = features_df.drop('cbwd')

            # Normalize the dataset
            features = self.normalize(features_df.to_numpy())

            # Define input and output
            self.X = torch.tensor(features[:-1], dtype=torch.float)
            self.y = torch.tensor(features[1:, -1:], dtype=torch.float)
            self.len = self.X.shape[0]

        except ImportError:
            print("Please install the 'ucimlrepo' package to use the AirQuality dataset.")

class STTREDatamodule(L.LightningDataModule):
    def __init__(self, data_dir: str = Config.DATA_DIR, batch_size: int = 32, seq_len: int = 60):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len

    def prepare_data(self):
        # Check if split files exist
        train_path = os.path.join(self.data_dir, 'uber_train.csv')
        val_path = os.path.join(self.data_dir, 'uber_val.csv')
        test_path = os.path.join(self.data_dir, 'uber_test.csv')
        original_path = os.path.join(self.data_dir, 'uber_stock.csv')

        if not all([os.path.exists(p) for p in [train_path, val_path, test_path]]):
            # Perform splitting
            df = pl.read_csv(original_path).sample(n=len(df), with_replacement=False, shuffle=True, seed=42)
            total = len(df)
            train_end = int(0.7 * total)
            val_end = train_end + int(0.15 * total)

            train_df = df[:train_end]
            val_df = df[train_end:val_end]
            test_df = df[val_end:]

            train_df.write_csv(train_path)
            val_df.write_csv(val_path)
            test_df.write_csv(test_path)

            print(f"Dataset split into train, val, test and saved to {self.data_dir}")

    def setup(self, stage=None):
        # Load datasets
        uber_train = Uber(os.path.join(self.data_dir, 'uber_train.csv'), seq_len=self.seq_len)
        uber_val = Uber(os.path.join(self.data_dir, 'uber_val.csv'), seq_len=self.seq_len)
        uber_test = Uber(os.path.join(self.data_dir, 'uber_test.csv'), seq_len=self.seq_len)

        self.train_dataset = uber_train
        self.valid_dataset = uber_val
        self.test_dataset = uber_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

def split_dataset(file_path, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
    # Load the dataset
    df = pl.read_csv(file_path)

    # Shuffle the dataset
    df = df.sample(n=len(df), with_replacement=False, shuffle=True, seed=seed)

    # Calculate split indices
    total = len(df)
    train_end = int(train_frac * total)
    val_end = train_end + int(val_frac * total)

    # Split the dataset
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    # Define output paths
    train_path = os.path.join(Config.DATA_DIR, 'uber_train.csv')
    val_path = os.path.join(Config.DATA_DIR, 'uber_val.csv')
    test_path = os.path.join(Config.DATA_DIR, 'uber_test.csv')

    # Save the splits
    train_df.write_csv(train_path)
    val_df.write_csv(val_path)
    test_df.write_csv(test_path)

    print(f"Dataset split into:\n- Train: {train_path}\n- Validation: {val_path}\n- Test: {test_path}")