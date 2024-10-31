import lightning as L
from torch.utils.data import DataLoader
import numpy as np

from .dataset import TimeSeriesDataset

class BaseDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, seq_len: int, columns: list, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.columns = columns
        self.batch_size = batch_size
        self.data = None
        
    def setup(self, stage=None):
        # Load and process data
        data = []
        with open(self.data_dir, 'r') as file:
            next(file)  # Skip header
            for line in file:
                try:
                    row = [float(line.split(',')[col]) for col in self.columns]
                    data.append(row)
                except ValueError:
                    continue
                    
        data = np.array(data)
        X = self.normalize(data[:, :-1])
        y = data[:, [-1]]
        
        # Split data into train/val/test
        train_size = int(0.7 * len(y))
        val_size = int(0.15 * len(y))
        
        self.train_data = (X[:train_size], y[:train_size])
        self.val_data = (X[train_size:train_size+val_size], 
                        y[train_size:train_size+val_size])
        self.test_data = (X[train_size+val_size:], y[train_size+val_size:])

    def train_dataloader(self):
        return DataLoader(self._create_dataset(*self.train_data), 
                         batch_size=self.batch_size, 
                         shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._create_dataset(*self.val_data), 
                         batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self._create_dataset(*self.test_data), 
                         batch_size=self.batch_size)
                         
    def _create_dataset(self, X, y):
        return TimeSeriesDataset(X, y, self.seq_len)

    @staticmethod
    def normalize(X):
        X = np.transpose(X)
        X_norm = [(x - np.min(x)) / (np.max(x) - np.min(x)) for x in X]
        return np.transpose(X_norm) 