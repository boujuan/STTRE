import lightning as L
from torch.utils.data import DataLoader
import numpy as np
import os

from .dataset import TimeSeriesDataset

class BaseDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, seq_len: int, columns: list, batch_size: int = 32):
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.seq_len = seq_len
        self.columns = columns
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        try:
            data = []
            with open(self.data_dir, 'r') as file:
                header = next(file)  # Read header to get column names
                next(file)  # Skip Ticker row
                next(file)  # Skip Date row
                
                for line in file:
                    values = line.strip().split(',')
                    if len(values) >= max(self.columns):  # Make sure we have enough columns
                        try:
                            # Convert only the columns we need
                            row = []
                            for col in self.columns:
                                val = float(values[col])
                                row.append(val)
                            data.append(row)
                        except (ValueError, IndexError):
                            continue
            
            if not data:
                raise ValueError("No valid data rows found")
            
            data = np.array(data)
            # Split into features and target (last column is Price)
            X = data[:, :-1]  # All columns except Price
            y = data[:, [-1]]  # Price column only
            
            # Normalize
            X = self.normalize(X)
            y = self.normalize(y)
            
            # Split data
            train_size = int(0.7 * len(y))
            val_size = int(0.15 * len(y))
            
            self.train_data = (X[:train_size], y[:train_size])
            self.val_data = (X[train_size:train_size+val_size], 
                            y[train_size:train_size+val_size])
            self.test_data = (X[train_size+val_size:], y[train_size+val_size:])
            
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def train_dataloader(self):
        return DataLoader(
            TimeSeriesDataset(self.train_data[0], self.train_data[1], self.seq_len),
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            TimeSeriesDataset(self.val_data[0], self.val_data[1], self.seq_len),
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            TimeSeriesDataset(self.test_data[0], self.test_data[1], self.seq_len),
            batch_size=self.batch_size
        )

    @staticmethod
    def normalize(X):
        X = np.array(X, dtype=np.float32)
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        return (X - min_vals) / (max_vals - min_vals)