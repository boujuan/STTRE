#!/usr/bin/env python
# coding: utf-8

# TODO:
# - Add my dataset
# - Turn into pytorch lightning for easier parallelization
# - Add decoder
# - Add parallelization (DistributedDataParallel)
# - Add dataloader for multiple datasets
# - Add automatic hyperparameter tuning (Population Based Training)

import warnings
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import polars as pl
import sys

warnings.filterwarnings('ignore', category=UserWarning, module='pandas.core.computation.expressions')

class Colors:
    # Regular colors
    BLUE = '\033[94m'      # Light/Bright Blue
    RED = '\033[91m'       # Light/Bright Red
    GREEN = '\033[92m'     # Light/Bright Green
    YELLOW = '\033[93m'    # Light/Bright Yellow
    CYAN = '\033[96m'      # Light/Bright Cyan
    MAGENTA = '\033[95m'   # Light/Bright Magenta
    
    # Bold colors
    BOLD_BLUE = '\033[1;34m'
    BOLD_GREEN = '\033[1;32m'
    BOLD_RED = '\033[1;31m'
    
    # Text style
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # End color
    ENDC = '\033[0m'
    
    # Emojis
    ROCKET = '🚀'
    HOURGLASS = '⌛'
    CHECK = '✅'
    CROSS = '❌'
    FIRE = '🔥'
    CHART = '📊'
    WARNING = '⚠️'
    BRAIN = '🧠'
    SAVE = '💾'
    STAR = '⭐'
    
    @classmethod
    def disable_colors(cls):
        """Disable colors if terminal doesn't support them"""
        for attr in dir(cls):
            if not attr.startswith('__') and isinstance(getattr(cls, attr), str):
                setattr(cls, attr, '')

    @staticmethod
    def supports_color():
        """Check if the terminal supports colors"""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

# Initialize colors based on terminal support
if not Colors.supports_color():
    Colors.disable_colors()

class Config:
    HOME = str(Path.home())
    SAVE_DIR = os.path.join(HOME, 'STTRE_outputs')
    MODEL_DIR = os.path.join(SAVE_DIR, 'models')
    PLOT_DIR = os.path.join(SAVE_DIR, 'plots')
    DATA_DIR = os.path.join(SAVE_DIR, 'data')

    @classmethod
    def create_directories(cls):
        for directory in [cls.SAVE_DIR, cls.MODEL_DIR, cls.PLOT_DIR, cls.DATA_DIR]:
            os.makedirs(directory, exist_ok=True)

class DeviceManager:
    _device_info_printed = False

    @classmethod
    def get_device(cls):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if not cls._device_info_printed:
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")
                cls._device_info_printed = True
        else:
            device = torch.device('cpu')
            if not cls._device_info_printed:
                print("GPU not available, using CPU")
                cls._device_info_printed = True
        return device

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
        x = np.transpose(self.X[idx:idx+self.seq_len])
        label = self.y[idx+self.seq_len+1]
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
            
            # Convert to numpy arrays and handle missing values
            X = features_df.to_numpy().astype(np.float32)
            y = beijing_pm2_5.data.targets.to_numpy().astype(np.float32)
            
            # Combine features and target
            data = np.column_stack((X, y))
            
            # Remove rows with missing values
            data = data[~np.isnan(data).any(axis=1)]
            
            self.seq_len = seq_len
            self.X = self.normalize(data[:, :-1])
            self.y = data[:, [-1]]
            self.len = len(self.y)
            
        except Exception as e:
            print(f"Error fetching UCI data: {str(e)}")
            raise

    def __len__(self):
        return self.len - self.seq_len - 1

    def __getitem__(self, idx):
        x = np.transpose(self.X[idx:idx+self.seq_len])
        label = self.y[idx+self.seq_len+1]
        return torch.tensor(x, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    def normalize(self, X):
        X = np.transpose(X)
        X_norm = [(x - np.min(x)) / (np.max(x) - np.min(x)) for x in X]
        return np.transpose(X_norm)

class Traffic(BaseDataset):
    def __init__(self, dir, seq_len=24):
        super().__init__(dir, seq_len, columns=range(8))

class AppliancesEnergy1(BaseDataset):
    def __init__(self, dir, seq_len=144):
        super().__init__(dir, seq_len, columns=range(26))

class AppliancesEnergy2(BaseDataset):
    def __init__(self, dir, seq_len=144):
        super().__init__(dir, seq_len, columns=range(26))

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, rel_emb):
        super(SelfAttention, self).__init__()
        self.device = DeviceManager.get_device()
        self.embed_size = embed_size
        self.heads = heads
        self.seq_len = seq_len
        self.module = module
        self.rel_emb = rel_emb
        modules = ['spatial', 'temporal', 'spatiotemporal', 'output']
        assert (module in modules), "Invalid module"

        if module in ['spatial', 'temporal']:
            self.head_dim = seq_len
            self.values = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32)
            self.keys = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32, device=self.device)
            self.queries = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32, device=self.device)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([self.heads, self.head_dim, self.embed_size], device=self.device))
        else:
            self.head_dim = embed_size // heads
            assert (self.head_dim * heads == embed_size), "Embed size not div by heads"
            self.values = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32)
            self.keys = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32, device=self.device)
            self.queries = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32, device=self.device)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([1, self.seq_len, self.head_dim], device=self.device))

        self.fc_out = nn.Linear(self.embed_size, self.embed_size, device=self.device)

    def forward(self, x):
        N, _, _ = x.shape

        if self.module in ['spatial', 'temporal']:
            values = self.values(x)
            keys = self.keys(x)
            queries = self.queries(x)
            values = values.reshape(N, self.seq_len, self.heads, self.embed_size)
            keys = keys.reshape(N, self.seq_len, self.heads, self.embed_size)
            queries = queries.reshape(N, self.seq_len, self.heads, self.embed_size)
        else:
            values, keys, queries = x, x, x
            values = values.reshape(N, self.seq_len, self.heads, self.head_dim)
            keys = keys.reshape(N, self.seq_len, self.heads, self.head_dim)
            queries = queries.reshape(N, self.seq_len, self.heads, self.head_dim)
            values = self.values(values)
            keys = self.keys(keys)
            queries = self.queries(queries)

        if self.rel_emb:
            QE = torch.matmul(queries.transpose(1, 2), self.E.transpose(1,2))
            QE = self._mask_positions(QE)
            S = self._skew(QE).contiguous().view(N, self.heads, self.seq_len, self.seq_len)
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len, device=self.device), 1)
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))
            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3) + S
        else:
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len, device=self.device), 1)
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))
            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3)

        if self.module in ['spatial', 'temporal']:
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len*self.heads, self.embed_size)
        else:
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len, self.heads*self.head_dim)

        z = self.fc_out(z)
        return z

    def _mask_positions(self, qe):
        L = qe.shape[-1]
        mask = torch.triu(torch.ones(L, L, device=self.device), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    def _skew(self, qe):
        padded_qe = F.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        return padded_qe[:,:,1:,:]

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, forward_expansion, rel_emb):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, seq_len, module, rel_emb=rel_emb)

        if module in ['spatial', 'temporal']:
            self.norm1 = nn.BatchNorm1d(seq_len*heads)
            self.norm2 = nn.BatchNorm1d(seq_len*heads)
        else:
            self.norm1 = nn.BatchNorm1d(seq_len)
            self.norm2 = nn.BatchNorm1d(seq_len)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.LeakyReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out

class Encoder(nn.Module):
    def __init__(self, seq_len, embed_size, num_layers, heads, device, forward_expansion, module, output_size=1, rel_emb=True):
        super(Encoder, self).__init__()
        self.module = module
        self.embed_size = embed_size
        self.device = device
        self.rel_emb = rel_emb
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, seq_len, module, forward_expansion=forward_expansion, rel_emb=rel_emb)
             for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
        out = self.fc_out(out)
        return out

class STTRE(nn.Module):
    def __init__(self, input_shape, output_size, embed_size, num_layers, forward_expansion, heads):
        super(STTRE, self).__init__()
        self.batch_size, self.num_var, self.seq_len = input_shape
        self.device = DeviceManager.get_device()
        self.num_elements = self.seq_len * self.num_var
        self.embed_size = embed_size
        self.element_embedding = nn.Linear(self.seq_len, embed_size*self.seq_len)
        self.pos_embedding = nn.Embedding(self.seq_len, embed_size)
        self.variable_embedding = nn.Embedding(self.num_var, embed_size)

        self.temporal = Encoder(seq_len=self.seq_len, embed_size=embed_size, num_layers=num_layers,
                                heads=self.num_var, device=self.device, forward_expansion=forward_expansion,
                                module='temporal', rel_emb=True)

        self.spatial = Encoder(seq_len=self.num_var, embed_size=embed_size, num_layers=num_layers,
                               heads=self.seq_len, device=self.device, forward_expansion=forward_expansion,
                               module='spatial', rel_emb=True)

        self.spatiotemporal = Encoder(seq_len=self.seq_len*self.num_var, embed_size=embed_size, num_layers=num_layers,
                                      heads=heads, device=self.device, forward_expansion=forward_expansion,
                                      module='spatiotemporal', rel_emb=True)

        self.fc_out1 = nn.Linear(embed_size, embed_size//2)
        self.fc_out2 = nn.Linear(embed_size//2, 1)
        self.out = nn.Linear((self.num_elements*3), output_size)

    def forward(self, x, dropout):
        batch_size = len(x)

        positions = torch.arange(0, self.seq_len).expand(batch_size, self.num_var, self.seq_len).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_temporal = self.element_embedding(x).reshape(batch_size, self.num_elements, self.embed_size)
        x_temporal = F.dropout(self.pos_embedding(positions) + x_temporal, dropout)

        x_spatial = torch.transpose(x, 1, 2).reshape(batch_size, self.num_var, self.seq_len)
        vars = torch.arange(0, self.num_var).expand(batch_size, self.seq_len, self.num_var).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_spatial = self.element_embedding(x_spatial).reshape(batch_size, self.num_elements, self.embed_size)
        x_spatial = F.dropout(self.variable_embedding(vars) + x_spatial, dropout)

        positions = torch.arange(0, self.seq_len).expand(batch_size, self.num_var, self.seq_len).reshape(batch_size, self.num_var* self.seq_len).to(self.device)
        x_spatio_temporal = self.element_embedding(x).reshape(batch_size, self.seq_len* self.num_var, self.embed_size)
        x_spatio_temporal = F.dropout(self.pos_embedding(positions) + x_spatio_temporal, dropout)

        out1 = self.temporal(x_temporal)
        out2 = self.spatial(x_spatial)
        out3 = self.spatiotemporal(x_spatio_temporal)
        out = torch.cat((out1, out2, out3), 1)
        out = self.fc_out1(out)
        out = F.leaky_relu(out)
        out = self.fc_out2(out)
        out = F.leaky_relu(out)
        out = torch.flatten(out, 1)
        out = self.out(out)

        return out

class ProgressBar:
    def __init__(self, initial_error, target_error=0, width=50):
        self.initial_error = initial_error
        self.target_error = target_error
        self.width = width
        self.best_error = initial_error
        
    def update(self, current_error):
        self.best_error = min(self.best_error, current_error)
        # Calculate progress (0 to 1) where 1 means error reduced to target
        progress = 1 - (self.best_error - self.target_error) / (self.initial_error - self.target_error)
        progress = max(0, min(1, progress))  # Clamp between 0 and 1
        
        # Create the progress bar
        filled_length = int(self.width * progress)
        bar = '█' * filled_length + '░' * (self.width - filled_length)
        
        # Calculate percentage
        percent = progress * 100
        
        return f'{Colors.BOLD_BLUE}Progress: |{Colors.GREEN}{bar}{Colors.BOLD_BLUE}| {percent:6.2f}% {Colors.ENDC}'

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.current_epoch = 0
        self.progress_bar = None

    def __call__(self, val_loss, model, path, epoch=None):
        # Initialize progress bar with first validation loss if not exists
        if self.progress_bar is None:
            self.progress_bar = ProgressBar(val_loss)
            
        self.current_epoch = epoch if epoch is not None else self.current_epoch + 1
        
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'\n{Colors.BOLD_GREEN}Epoch {self.current_epoch}:{Colors.ENDC} {Colors.RED}Validation loss increased ({self.best_loss:.6f} --> {val_loss:.6f}) {Colors.CROSS}{Colors.ENDC} {Colors.BOLD_RED}[{self.counter}/{self.patience}]{Colors.ENDC}{Colors.WARNING}')
                print(self.progress_bar.update(val_loss))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose and (self.best_loss - val_loss) > self.min_delta:
                print(f'\n{Colors.BOLD_GREEN}Epoch {self.current_epoch}:{Colors.ENDC} {Colors.GREEN}Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}) {Colors.FIRE}{Colors.ENDC}')
                # print(f'{Colors.BLUE}Saving model... {Colors.SAVE}{Colors.ENDC}')
                print(self.progress_bar.update(val_loss))
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        # if self.verbose and (self.val_loss_min - val_loss) > self.min_delta:
        #     print(f'Saving model ...')
        full_path = os.path.join(Config.MODEL_DIR, path)
        torch.save(model.state_dict(), full_path)
        self.val_loss_min = val_loss

class Plotter:
    @staticmethod
    def plot_metrics(train_metrics, val_metrics, metric_names, dataset):
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.5)
        
        if not hasattr(Plotter.plot_metrics, 'fig'):
            Plotter.plot_metrics.fig, Plotter.plot_metrics.axes = plt.subplots(1, len(metric_names), 
                                                                               figsize=(20, 7), 
                                                                               dpi=300)
        
        all_data = []
        for metric_name in metric_names:
            epochs = list(range(1, len(train_metrics[metric_name]) + 1))
            
            all_data.append(pl.DataFrame({
                'Epoch': epochs,
                'Value': train_metrics[metric_name],
                'Type': ['Train'] * len(epochs),
                'Metric': [metric_name] * len(epochs)
            }))
            
            all_data.append(pl.DataFrame({
                'Epoch': epochs,
                'Value': val_metrics[metric_name],
                'Type': ['Validation'] * len(epochs),
                'Metric': [metric_name] * len(epochs)
            }))
        
        df = pl.concat(all_data)
        
        for i, metric_name in enumerate(metric_names):
            Plotter.plot_metrics.axes[i].clear()
            
            metric_data = df.filter(pl.col('Metric') == metric_name)
            
            sns.lineplot(data=metric_data, x='Epoch', y='Value', hue='Type', 
                        ax=Plotter.plot_metrics.axes[i],
                        palette=['#2ecc71', '#e74c3c'],
                        linewidth=2.5)
            
            Plotter.plot_metrics.axes[i].set_title(metric_name, pad=20, fontsize=16, fontweight='bold')
            Plotter.plot_metrics.axes[i].set_xlabel('Epoch', fontsize=12)
            Plotter.plot_metrics.axes[i].set_ylabel(metric_name, fontsize=12)
            Plotter.plot_metrics.axes[i].legend(title=None, fontsize=10)
            Plotter.plot_metrics.axes[i].tick_params(axis='both', which='major', labelsize=10)
        
        Plotter.plot_metrics.fig.tight_layout(pad=3.0)
        Plotter.plot_metrics.fig.savefig(os.path.join(Config.PLOT_DIR, f'{dataset}_metrics_latest.png'), 
                                        bbox_inches='tight',
                                        facecolor='white',
                                        edgecolor='none')

class STTRETrainer:
    def __init__(self, model_params, train_params):
        self.model_params = model_params
        self.train_params = train_params
        self.device = DeviceManager.get_device()

    def prepare_data(self, dataset_class, dir):
        try:
            dataset = dataset_class(dir)
            if len(dataset) == 0:
                raise ValueError(f"Dataset is empty after processing. Please check the file: {dir}")
            
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(self.train_params['TEST_SPLIT'] * dataset_size))
            
            np.random.seed(42)
            np.random.shuffle(indices)
            train_indices, test_indices = indices[split:], indices[:split]
            
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            
            train_dataloader = DataLoader(
                dataset, 
                batch_size=self.train_params['batch_size'],
                sampler=train_sampler,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True
            )
            test_dataloader = DataLoader(
                dataset, 
                batch_size=self.train_params['batch_size'],
                sampler=test_sampler,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True
            )
            
            return train_dataloader, test_dataloader

        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            return None, None

    def train(self, train_dataloader, test_dataloader, dataset_name):
        inputs, _ = next(iter(train_dataloader))
        model = STTRE(
            inputs.shape, 
            1, 
            embed_size=self.model_params['embed_size'],
            num_layers=self.model_params['num_layers'],
            forward_expansion=self.model_params['forward_expansion'],
            heads=self.model_params['heads']
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_params['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=20
        )
        loss_fn = nn.MSELoss()

        early_stopping = EarlyStopping(patience=20, verbose=True)
        history = {
            'train': {'MSE': [], 'MAE': [], 'MAPE': []},
            'val': {'MSE': [], 'MAE': [], 'MAPE': []}
        }

        metrics = {
            'train': {
                'mse': MeanSquaredError().to(self.device),
                'mae': MeanAbsoluteError().to(self.device),
                'mape': MeanAbsolutePercentageError().to(self.device)
            },
            'val': {
                'mse': MeanSquaredError().to(self.device),
                'mae': MeanAbsoluteError().to(self.device),
                'mape': MeanAbsolutePercentageError().to(self.device)
            }
        }

        # Initialize progress bar with first batch loss
        inputs, labels = next(iter(train_dataloader))
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = model(inputs, self.train_params['dropout'])
        initial_loss = loss_fn(outputs, labels).item()
        progress_bar = ProgressBar(initial_loss)
        
        for epoch in range(self.train_params['NUM_EPOCHS']):
            model.train()
            train_loss = 0
            num_batches = 0
            
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                
                outputs = model(inputs, self.train_params['dropout'])
                loss = loss_fn(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                for metric in metrics['train'].values():
                    metric.update(outputs, labels)

            train_loss = train_loss / num_batches

            model.eval()
            val_loss = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs, 0)
                    val_loss += loss_fn(outputs, labels).item()
                    num_val_batches += 1
                    
                    for metric in metrics['val'].values():
                        metric.update(outputs, labels)

            val_loss = val_loss / num_val_batches

            current_metrics = {
                'train': {},
                'val': {}
            }
            
            for phase in ['train', 'val']:
                for name, metric in metrics[phase].items():
                    value = metric.compute()
                    current_metrics[phase][name] = value
                    history[phase][name.upper()].append(value.item())
                    metric.reset()

            scheduler.step(val_loss)
            
            early_stopping(val_loss, model, f'best_model_{dataset_name}.pth', epoch + 1)
            
            # After validation phase, update and display progress
            # if epoch % 5 == 0:
            #     print(f'{Colors.BOLD_BLUE}Epoch {epoch+1}/{self.train_params["NUM_EPOCHS"]} {Colors.HOURGLASS}{Colors.ENDC}')
            #     print(f'{Colors.CYAN}Train Loss: {train_loss:.4f}, '
            #           f'MSE: {current_metrics["train"]["mse"]:.4f}, '
            #           f'MAE: {current_metrics["train"]["mae"]:.4f}, '
            #           f'MAPE: {current_metrics["train"]["mape"]:.4f} {Colors.CHART}{Colors.ENDC}')
            #     print(f'{Colors.YELLOW}Val Loss: {val_loss:.4f}, '
            #           f'MSE: {current_metrics["val"]["mse"]:.4f}, '
            #           f'MAE: {current_metrics["val"]["mae"]:.4f}, '
            #           f'MAPE: {current_metrics["val"]["mape"]:.4f} {Colors.BRAIN}{Colors.ENDC}')
            #     print(progress_bar.update(val_loss))
            #     print()  # Add empty line for readability

            if early_stopping.early_stop:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        Plotter.plot_metrics(history['train'], history['val'], ['MSE', 'MAE', 'MAPE'], dataset_name)
        
        return history

    def validate(self, model_path, test_dataloader):
        inputs, _ = next(iter(test_dataloader))
        
        # Create model for initial metrics (untrained)
        initial_model = STTRE(
            inputs.shape, 
            1, 
            embed_size=self.model_params['embed_size'],
            num_layers=self.model_params['num_layers'],
            forward_expansion=self.model_params['forward_expansion'],
            heads=self.model_params['heads']
        ).to(self.device)
        
        # Create model for final metrics (trained)
        trained_model = STTRE(
            inputs.shape, 
            1, 
            embed_size=self.model_params['embed_size'],
            num_layers=self.model_params['num_layers'],
            forward_expansion=self.model_params['forward_expansion'],
            heads=self.model_params['heads']
        ).to(self.device)
        
        # Load trained weights
        trained_model.load_state_dict(torch.load(model_path, weights_only=True))
        trained_model.eval()

        metrics = {
            'mse': MeanSquaredError().to(self.device),
            'mae': MeanAbsoluteError().to(self.device),
            'mape': MeanAbsolutePercentageError().to(self.device)
        }

        # Capture initial metrics with untrained model
        initial_metrics = {}
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = initial_model(inputs, 0)
                
                for name, metric in metrics.items():
                    metric.update(outputs, labels)
                
            for name, metric in metrics.items():
                initial_metrics[name] = metric.compute().item()
                metric.reset()
        
        # Perform validation with trained model
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = trained_model(inputs, 0)
                
                for metric in metrics.values():
                    metric.update(outputs, labels)

        results = {name: metric.compute().item() for name, metric in metrics.items()}
        return results, initial_metrics

def format_validation_results(results, initial_metrics):
    """Creates a pretty formatted string for validation results"""
    # Header
    output = [
        f"\n{Colors.BOLD_BLUE}{'='*60}{Colors.ENDC}",
        f"{Colors.BOLD_BLUE}{'Final Validation Results':^60}{Colors.ENDC}",
        f"{Colors.BOLD_BLUE}{'='*60}{Colors.ENDC}\n"
    ]
    
    # Metrics with their corresponding emojis and descriptions
    metrics_info = {
        'mse': ('MSE', '📊', 'Mean Squared Error'),
        'mae': ('MAE', '📏', 'Mean Absolute Error'),
        'mape': ('MAPE', '📈', 'Mean Absolute Percentage Error')
    }
    
    # Add each metric with formatting
    for metric, (short_name, emoji, full_name) in metrics_info.items():
        final_value = results[metric]
        initial_value = initial_metrics[metric]
        
        # Calculate improvement percentage
        if initial_value > 0:  # Prevent division by zero
            improvement = (initial_value - final_value) / initial_value
            improvement = max(0, min(1, improvement))  # Clamp between 0 and 1
        else:
            improvement = 0
            
        # Create visual bar based on improvement
        max_bars = 20
        bars = '█' * int(improvement * max_bars) + '░' * (max_bars - int(improvement * max_bars))
        
        # Calculate percentage improvement
        percentage = improvement * 100
        
        # Color code based on improvement
        if percentage > 50:
            value_color = Colors.BOLD_GREEN
        elif percentage > 25:
            value_color = Colors.BOLD_BLUE
        else:
            value_color = Colors.BOLD_RED
            
        output.extend([
            f"{Colors.CYAN}{emoji} {short_name}:{Colors.ENDC}",
            f"{Colors.YELLOW}├─ Initial: {initial_value:.6f}{Colors.ENDC}",
            f"{Colors.YELLOW}├─ Final:   {value_color}{final_value:.6f}{Colors.ENDC}",
            f"{Colors.GREEN}└─ Improvement: |{bars}| {percentage:.1f}%{Colors.ENDC}\n"
        ])
    
    # Footer
    output.extend([
        f"{Colors.BOLD_BLUE}{'='*60}{Colors.ENDC}",
        f"{Colors.MAGENTA}Analysis completed successfully! {Colors.STAR}{Colors.ENDC}\n"
    ])
    
    return '\n'.join(output)

########################################################################################
###################################### MAIN ############################################
########################################################################################

def main(mode='both'):
    """
    Run the STTRE model in different modes.
    Args:
        mode (str): One of 'train', 'validate', or 'both'
    """
    if mode not in ['train', 'validate', 'both']:
        raise ValueError("Mode must be one of: 'train', 'validate', 'both'")

    Config.create_directories()
    
    '''
    MODEL PARAMETERS
    
    [UBER]
    embed_size: 64
    heads: 8
    num_layers: 4
    forward_expansion: 2
    dropout: 0.15
    lr: 0.00005
    batch_size: 256
    NUM_EPOCHS: 1000
    TEST_SPLIT: 0.2
    '''
    model_params = {
        'embed_size': 32, # Default: 32
        'heads': 4, # Default: 4
        'num_layers': 3, # Default: 3
        'forward_expansion': 1 # Default: 1
    }

    train_params = {
        'dropout': 0.2, # Default: 0.2
        'lr': 0.0001, # Default: 0.0001
        'batch_size': 256, # Default: 512
        'NUM_EPOCHS': 1320, # Default: 1000
        'TEST_SPLIT': 0.3 # Default: 0.3
    }

    trainer = STTRETrainer(model_params, train_params)

    datasets = {
        # 'Uber': (Uber, os.path.join(Config.DATA_DIR, 'uber_stock.csv')),
        'AirQuality': (AirQuality, None),
        # 'IstanbulStock': (IstanbulStock, os.path.join(Config.DATA_DIR, 'istanbul_stock.csv')),
        # 'Traffic': (Traffic, os.path.join(Config.DATA_DIR, 'traffic.csv')),
        # 'AppliancesEnergy1': (AppliancesEnergy1, os.path.join(Config.DATA_DIR, 'appliances_energy1.csv')),
        # 'AppliancesEnergy2': (AppliancesEnergy2, os.path.join(Config.DATA_DIR, 'appliances_energy2.csv'))
    }

    for dataset_name, (dataset_class, data_path) in datasets.items():
        print(f"\n{Colors.BOLD_GREEN}Processing {dataset_name} dataset {Colors.ROCKET}{Colors.ENDC}")
        train_dataloader, test_dataloader = trainer.prepare_data(dataset_class, data_path)
        
        if train_dataloader is None or test_dataloader is None:
            print(f"Skipping {dataset_name} due to data preparation error")
            continue

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Training phase
            if mode in ['train', 'both']:
                print(f"{Colors.MAGENTA}Training {dataset_name}... {Colors.BRAIN}{Colors.ENDC}")
                history = trainer.train(train_dataloader, test_dataloader, dataset_name)
                print(f"{Colors.BOLD_GREEN}Completed training on {dataset_name} dataset {Colors.CHECK}{Colors.ENDC}")

            # Validation phase
            if mode in ['validate', 'both']:
                print(f"{Colors.MAGENTA}Validating {dataset_name}... {Colors.CHART}{Colors.ENDC}")
                model_path = os.path.join(Config.MODEL_DIR, f'best_model_{dataset_name}.pth')
                if not os.path.exists(model_path):
                    print(f"No trained model found for {dataset_name} at {model_path}")
                    continue
                    
                validation_results, initial_metrics = trainer.validate(model_path, test_dataloader)
                print(format_validation_results(validation_results, initial_metrics))

        except Exception as e:
            print(f"{Colors.RED}Error in experiment with dataset {dataset_name}: {str(e)} {Colors.CROSS}{Colors.ENDC}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{Colors.BOLD_GREEN}All experiments completed! {Colors.CHECK}{Colors.ENDC}")

if __name__ == "__main__":
    # You can change this to 'train', 'validate', or 'both'
    main(mode='both')

