#!/usr/bin/env python
# coding: utf-8

# TODO:
# - Add my dataset
# - Turn into pytorch lightning for easier parallelization ✅
# - Add decoder
# - Add parallelization (DistributedDataParallel) ✅
# - Add dataloader for multiple datasets
# - Add automatic hyperparameter tuning (Population Based Training)

import warnings
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import polars as pl
import sys
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError

warnings.filterwarnings('ignore', category=UserWarning, module='pandas.core.computation.expressions')

torch.set_float32_matmul_precision('medium')

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
            # Ensure train and val metrics have the same length
            min_len = min(len(train_metrics[metric_name]), len(val_metrics[metric_name]))
            epochs = list(range(1, min_len + 1))
            
            # Truncate metrics to the same length
            train_values = train_metrics[metric_name][:min_len]
            val_values = val_metrics[metric_name][:min_len]
            
            train_df = pl.DataFrame({
                'Epoch': epochs,
                'Value': train_values,
                'Type': ['Train'] * min_len,
                'Metric': [metric_name] * min_len
            })
            
            val_df = pl.DataFrame({
                'Epoch': epochs,
                'Value': val_values,
                'Type': ['Validation'] * min_len,
                'Metric': [metric_name] * min_len
            })
            
            all_data.extend([train_df, val_df])
        
        try:
            df = pl.concat(all_data)
            
            for i, metric_name in enumerate(metric_names):
                ax = Plotter.plot_metrics.axes[i] if len(metric_names) > 1 else Plotter.plot_metrics.axes
                ax.clear()
                
                metric_data = df.filter(pl.col('Metric') == metric_name)
                
                # Convert to pandas for seaborn compatibility
                metric_data_pd = metric_data.to_pandas()
                
                sns.lineplot(data=metric_data_pd, x='Epoch', y='Value', hue='Type', 
                           ax=ax,
                           palette=['#2ecc71', '#e74c3c'],
                           linewidth=2.5)
                
                ax.set_title(metric_name, pad=20, fontsize=16, fontweight='bold')
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel(metric_name, fontsize=12)
                ax.legend(title=None, fontsize=10)
                ax.tick_params(axis='both', which='major', labelsize=10)
            
            Plotter.plot_metrics.fig.tight_layout(pad=3.0)
            Plotter.plot_metrics.fig.savefig(os.path.join(Config.PLOT_DIR, f'{dataset}_metrics_latest.png'), 
                                           bbox_inches='tight',
                                           facecolor='white',
                                           edgecolor='none')
            
        except Exception as e:
            print(f"Error in plotting: {str(e)}")
            # Don't let plotting errors stop the training process
            pass

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
    def __init__(self, embed_size, heads, seq_len, module, rel_emb, device):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.seq_len = seq_len
        self.module = module
        self.rel_emb = rel_emb
        self.device = device
        modules = ['spatial', 'temporal', 'spatiotemporal', 'output']
        assert (module in modules), "Invalid module"

        if module in ['spatial', 'temporal']:
            self.head_dim = seq_len
            self.values = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32)
            self.keys = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32)
            self.queries = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([self.heads, self.head_dim, self.embed_size]).to(self.device))
        else:
            self.head_dim = embed_size // heads
            assert (self.head_dim * heads == embed_size), "Embed size not div by heads"
            self.values = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32)
            self.keys = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32)
            self.queries = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([1, self.seq_len, self.head_dim]).to(self.device))

        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

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
            # Create mask on the same device as input tensor
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len, device=x.device), 1)
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))
            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3) + S
        else:
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            # Create mask on the same device as input tensor
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len, device=x.device), 1)
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
        # Create mask on the same device as input tensor
        mask = torch.triu(torch.ones(L, L, device=qe.device), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    def _skew(self, qe):
        padded_qe = F.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        return padded_qe[:,:,1:,:]

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, forward_expansion, rel_emb, device):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, seq_len, module, rel_emb=rel_emb, device=device)

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
            [TransformerBlock(embed_size, heads, seq_len, module, forward_expansion=forward_expansion, rel_emb=rel_emb, device=device)
             for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
        out = self.fc_out(out)
        return out

class LitSTTRE(L.LightningModule):
    def __init__(self, input_shape, output_size, model_params, train_params):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size, self.num_var, self.seq_len = input_shape
        self.num_elements = self.seq_len * self.num_var
        self.embed_size = model_params['embed_size']
        self.train_params = train_params
        
        # Model components
        self.element_embedding = nn.Linear(self.seq_len, model_params['embed_size']*self.seq_len)
        self.pos_embedding = nn.Embedding(self.seq_len, model_params['embed_size'])
        self.variable_embedding = nn.Embedding(self.num_var, model_params['embed_size'])
        
        # Encoder components
        self.temporal = Encoder(
            seq_len=self.seq_len,
            embed_size=model_params['embed_size'],
            num_layers=model_params['num_layers'],
            heads=self.num_var,
            device=self.device,
            forward_expansion=model_params['forward_expansion'],
            module='temporal',
            rel_emb=True
        )
        
        self.spatial = Encoder(
            seq_len=self.num_var,
            embed_size=model_params['embed_size'],
            num_layers=model_params['num_layers'],
            heads=self.seq_len,
            device=self.device,
            forward_expansion=model_params['forward_expansion'],
            module='spatial',
            rel_emb=True
        )
        
        self.spatiotemporal = Encoder(
            seq_len=self.seq_len*self.num_var,
            embed_size=model_params['embed_size'],
            num_layers=model_params['num_layers'],
            heads=model_params['heads'],
            device=self.device,
            forward_expansion=model_params['forward_expansion'],
            module='spatiotemporal',
            rel_emb=True
        )
        
        # Output layers
        self.fc_out1 = nn.Linear(model_params['embed_size'], model_params['embed_size']//2)
        self.fc_out2 = nn.Linear(model_params['embed_size']//2, 1)
        self.out = nn.Linear((self.num_elements*3), output_size)
        
        # Initialize metrics
        metrics = ['mse', 'mae', 'mape']
        for split in ['train', 'val']:
            for metric in metrics:
                metric_class = {
                    'mse': MeanSquaredError,
                    'mae': MeanAbsoluteError,
                    'mape': MeanAbsolutePercentageError
                }[metric]
                setattr(self, f'{split}_{metric}', metric_class())
        
        # Initialize test metrics
        self.test_mse = None
        self.test_mae = None
        self.test_mape = None
        
        # Initialize training history
        self.training_history = {
            'train': {'MSE': [], 'MAE': [], 'MAPE': []},
            'val': {'MSE': [], 'MAE': [], 'MAPE': []}
        }
        
        # Track whether metrics have been updated
        self.metrics_updated = False

    def forward(self, x):
        batch_size = len(x)
        
        # Temporal embedding
        positions = torch.arange(0, self.seq_len).expand(batch_size, self.num_var, self.seq_len).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_temporal = self.element_embedding(x).reshape(batch_size, self.num_elements, self.embed_size)
        x_temporal = F.dropout(self.pos_embedding(positions) + x_temporal, self.train_params['dropout'] if self.training else 0)
        
        # Spatial embedding
        x_spatial = torch.transpose(x, 1, 2).reshape(batch_size, self.num_var, self.seq_len)
        vars = torch.arange(0, self.num_var).expand(batch_size, self.seq_len, self.num_var).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_spatial = self.element_embedding(x_spatial).reshape(batch_size, self.num_elements, self.embed_size)
        x_spatial = F.dropout(self.variable_embedding(vars) + x_spatial, self.train_params['dropout'] if self.training else 0)
        
        # Spatiotemporal embedding
        x_spatio_temporal = self.element_embedding(x).reshape(batch_size, self.seq_len* self.num_var, self.embed_size)
        x_spatio_temporal = F.dropout(self.pos_embedding(positions) + x_spatio_temporal, self.train_params['dropout'] if self.training else 0)
        
        # Process through encoders
        out1 = self.temporal(x_temporal)
        out2 = self.spatial(x_spatial)
        out3 = self.spatiotemporal(x_spatio_temporal)
        
        # Final processing
        out = torch.cat((out1, out2, out3), 1)
        out = F.leaky_relu(self.fc_out1(out))
        out = F.leaky_relu(self.fc_out2(out))
        out = torch.flatten(out, 1)
        out = self.out(out)
        
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        
        # Update metrics
        self.train_mse(y_hat, y)
        self.train_mae(y_hat, y)
        self.train_mape(y_hat, y)
        self.metrics_updated = True
        
        # Log metrics with sync_dist=True for proper multi-GPU logging
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_mse', self.train_mse, prog_bar=True, sync_dist=True)
        self.log('train_mae', self.train_mae, prog_bar=True, sync_dist=True)
        self.log('train_mape', self.train_mape, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        
        # Update metrics
        self.val_mse(y_hat, y)
        self.val_mae(y_hat, y)
        self.val_mape(y_hat, y)
        
        # Log metrics with sync_dist=True
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        self.log('val_mse', self.val_mse, prog_bar=True, sync_dist=True)
        self.log('val_mae', self.val_mae, prog_bar=True, sync_dist=True)
        self.log('val_mape', self.val_mape, prog_bar=True, sync_dist=True)
        
        return val_loss

    def test_step(self, batch, batch_idx):
        if not hasattr(self, 'test_metrics_initialized'):
            self.test_mse = MeanSquaredError().to(self.device)
            self.test_mae = MeanAbsoluteError().to(self.device)
            self.test_mape = MeanAbsolutePercentageError().to(self.device)
            self.test_metrics_initialized = True
        
        x, y = batch
        y_hat = self(x)
        test_loss = F.mse_loss(y_hat, y)
        
        # Log with sync_dist=True
        self.log('test_loss', test_loss, sync_dist=True)
        self.log('test_mse', self.test_mse(y_hat, y), sync_dist=True)
        self.log('test_mae', self.test_mae(y_hat, y), sync_dist=True)
        self.log('test_mape', self.test_mape(y_hat, y), sync_dist=True)
        
        return test_loss

    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        # Only update history if metrics have been computed
        if not self.metrics_updated:
            return
            
        # Get current metrics
        metrics = {
            'train': {
                'MSE': float(self.trainer.callback_metrics.get('train_mse', 0)),
                'MAE': float(self.trainer.callback_metrics.get('train_mae', 0)),
                'MAPE': float(self.trainer.callback_metrics.get('train_mape', 0))
            },
            'val': {
                'MSE': float(self.trainer.callback_metrics.get('val_mse', 0)),
                'MAE': float(self.trainer.callback_metrics.get('val_mae', 0)),
                'MAPE': float(self.trainer.callback_metrics.get('val_mape', 0))
            }
        }
        
        # Update history
        for split in ['train', 'val']:
            for metric in ['MSE', 'MAE', 'MAPE']:
                self.training_history[split][metric].append(metrics[split][metric])
        
        try:
            # Plot metrics at the end of each epoch
            Plotter.plot_metrics(
                self.training_history['train'],
                self.training_history['val'],
                ['MSE', 'MAE', 'MAPE'],
                self.trainer.logger.name
            )
        except Exception as e:
            print(f"Warning: Could not plot metrics: {str(e)}")
            pass

    def on_train_epoch_start(self):
        """Reset metrics at the start of each epoch"""
        self.train_mse.reset()
        self.train_mae.reset()
        self.train_mape.reset()
        self.val_mse.reset()
        self.val_mae.reset()
        self.val_mape.reset()
        self.metrics_updated = False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_params['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=20
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

class STTREDataModule(L.LightningDataModule):
    def __init__(self, dataset_class, data_path, batch_size, test_split=0.2, val_split=0.1):
        super().__init__()
        self.dataset_class = dataset_class
        self.data_path = data_path
        self.batch_size = batch_size
        self.test_split = test_split
        self.val_split = val_split

    def setup(self, stage=None):
        try:
            # Create full dataset
            full_dataset = self.dataset_class(self.data_path)
            
            if len(full_dataset) == 0:
                raise ValueError(f"Dataset is empty after processing. Please check the file: {self.data_path}")
            
            # Calculate lengths
            dataset_size = len(full_dataset)
            test_size = int(self.test_split * dataset_size)
            val_size = int(self.val_split * dataset_size)
            train_size = dataset_size - test_size - val_size
            
            # Split dataset
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                full_dataset, 
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            print(f"{Colors.BOLD_BLUE}Dataset splits:{Colors.ENDC}")
            print(f"{Colors.CYAN}Training samples: {train_size}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Validation samples: {val_size}{Colors.ENDC}")
            print(f"{Colors.GREEN}Test samples: {test_size}{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error preparing data: {str(e)}{Colors.CROSS}{Colors.ENDC}")
            raise

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

def cleanup_old_checkpoints(model_dir, dataset_name, keep_top_k=3):
    """Clean up old checkpoints, keeping only the top k best models."""
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith(f'sttre-{dataset_name.lower()}-') and f.endswith('.ckpt')]
    if len(checkpoints) > keep_top_k:
        # Sort checkpoints by validation loss (extracted from filename)
        checkpoints.sort(key=lambda x: float(x.split('val_loss=')[1].split('.ckpt')[0]))
        # Remove all but the top k checkpoints
        for checkpoint in checkpoints[keep_top_k:]:
            os.remove(os.path.join(model_dir, checkpoint))

def train_sttre(dataset_class, data_path, model_params, train_params):
    """Train the STTRE model using the specified dataset."""
    Config.create_directories()
    
    # Initialize data module
    data_module = STTREDataModule(
        dataset_class=dataset_class,
        data_path=data_path,
        batch_size=train_params['batch_size']
    )
    
    # Setup data module to get input shape
    data_module.setup()
    sample_batch = next(iter(data_module.train_dataloader()))
    input_shape = sample_batch[0].shape
    
    # Initialize model
    model = LitSTTRE(
        input_shape=input_shape,
        output_size=model_params['output_size'],
        model_params=model_params,
        train_params=train_params
    )
    
    # Setup callbacks
    callbacks = [
        RichProgressBar(
            theme=RichProgressBarTheme(
                description="white",
                progress_bar="#6206E0",
                progress_bar_finished="#6206E0",
                progress_bar_pulse="#6206E0",
                batch_progress="white",
                time="grey54",
                processing_speed="grey70",
                metrics="white"
            ),
            leave=True
        ),
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=Config.MODEL_DIR,
            filename=f'sttre-{dataset_class.__name__.lower()}-' + '{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=train_params.get('patience', 20),
            mode='min'
        )
    ]
    
    # Logger setup
    logger = TensorBoardLogger(
        save_dir=Config.SAVE_DIR,
        name=dataset_class.__name__.lower(),
        default_hp_metric=False
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=train_params['epochs'],
        accelerator='auto',
        devices='auto',
        strategy='ddp_find_unused_parameters_true',
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=train_params.get('gradient_clip', 1.0),
        precision=train_params.get('precision', 32),
        accumulate_grad_batches=train_params.get('accumulate_grad_batches', 1),
        log_every_n_steps=1,
        enable_progress_bar=True
    )

    try:
        print(f"\n{Colors.BOLD_GREEN}Starting training for {dataset_class.__name__} dataset {Colors.ROCKET}{Colors.ENDC}")
        trainer.fit(model, data_module)
        print(f"{Colors.BOLD_GREEN}Training completed! {Colors.CHECK}{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD_BLUE}Starting testing... {Colors.CHART}{Colors.ENDC}")
        test_results = trainer.test(model, datamodule=data_module)
        print(f"{Colors.BOLD_GREEN}Testing completed! {Colors.CHECK}{Colors.ENDC}")
        
        return model, trainer, test_results
        
    except Exception as e:
        print(f"{Colors.RED}Error during training: {str(e)} {Colors.CROSS}{Colors.ENDC}")
        raise

def test_sttre(dataset_class, data_path, model_params, train_params, checkpoint_path):
    """
    Test the STTRE model using a saved checkpoint.
    """
    Config.create_directories()
    
    # Initialize data module
    data_module = STTREDataModule(
        dataset_class=dataset_class,
        data_path=data_path,
        batch_size=train_params['batch_size'],
        test_split=train_params.get('test_split', 0.2),
        val_split=train_params.get('val_split', 0.1)
    )

    # Setup data module to get input shape
    data_module.setup()
    sample_batch = next(iter(data_module.train_dataloader()))
    input_shape = sample_batch[0].shape

    # Logger
    logger = TensorBoardLogger(
        save_dir=Config.SAVE_DIR,
        name=dataset_class.__name__.lower(),
        default_hp_metric=False
    )

    # Load model from checkpoint
    model = LitSTTRE.load_from_checkpoint(
        checkpoint_path,
        input_shape=input_shape,
        output_size=model_params['output_size'],
        model_params=model_params,
        train_params=train_params
    )

    # Testing trainer
    test_trainer = L.Trainer(
        accelerator='gpu',
        devices=[0],
        logger=logger,
        enable_progress_bar=True,
        strategy='auto'
    )

    # Move model to GPU and test
    model = model.cuda()
    test_results = test_trainer.test(model, datamodule=data_module, verbose=True)
    print(f"{Colors.BOLD_GREEN}Testing completed! {Colors.CHECK}{Colors.ENDC}")

    return model, test_results

if __name__ == "__main__":
    checkpoint_path = os.path.join(Config.MODEL_DIR, 'sttre-uber-epoch=519-val_loss=6.46.ckpt')
    
    model_params = {
        'embed_size': 32,
        'num_layers': 3,
        'heads': 4,
        'forward_expansion': 1,
        'output_size': 1
    }

    train_params = {
        'batch_size': 256,
        'epochs': 1000,
        'lr': 0.0001,
        'dropout': 0.2,
        'patience': 20,
        'gradient_clip': 1.0,
        'precision': 32,
        'accumulate_grad_batches': 1,
        'test_split': 0.2,
        'val_split': 0.1
    }

    datasets = {
        # 'AirQuality': (AirQuality, None),
        'Uber': (Uber, os.path.join(Config.DATA_DIR, 'uber_stock.csv')),
        # 'IstanbulStock': (IstanbulStock, os.path.join(Config.DATA_DIR, 'istanbul_stock.csv')),
        # 'Traffic': (Traffic, os.path.join(Config.DATA_DIR, 'traffic.csv')),
        # 'AppliancesEnergy1': (AppliancesEnergy1, os.path.join(Config.DATA_DIR, 'appliances_energy1.csv')),
        # 'AppliancesEnergy2': (AppliancesEnergy2, os.path.join(Config.DATA_DIR, 'appliances_energy2.csv'))
    }

    trainer = None
    for dataset_name, (dataset_class, data_path) in datasets.items():
        try:
            model, trainer, test_results = train_sttre(dataset_class, data_path, model_params, train_params)
            print(f"{Colors.BOLD_GREEN}Completed {dataset_name} dataset {Colors.CHECK}{Colors.ENDC}")
            
            # model, test_results = test_sttre(
            #     dataset_class, 
            #     data_path, 
            #     model_params, 
            #     train_params,
            #     checkpoint_path
            # )
            # print(f"{Colors.BOLD_GREEN}Completed testing {dataset_name} dataset {Colors.CHECK}{Colors.ENDC}")

            
        except Exception as e:
            print(f"{Colors.RED}Error processing {dataset_name}: {str(e)} {Colors.CROSS}{Colors.ENDC}")
            continue

    print(f"\n{Colors.BOLD_GREEN}All experiments completed! {Colors.CHECK}{Colors.ENDC}")