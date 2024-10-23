#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for HPC
import matplotlib.pyplot as plt
# import yfinance as yf

# CUDA setup
USE_CUDA = True
try:
    device = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("GPU not available, using CPU")
except Exception as e:
    print(f"Error setting up CUDA: {e}")
    device = torch.device('cpu')
print(f"Using device: {device}")

# Dataset classes
class Uber(Dataset):
    def __init__(self, dir, seq_len=60):
        self.seq_len = seq_len
        # Skip first 3 rows (headers) and select specific columns
        # usecols: High, Low, Open, Close, Volume
        data = np.loadtxt(dir, delimiter=',', skiprows=3, usecols=(3,4,5,2,6), dtype=float)
        self.X = self.normalize(data)
        self.y = data[:, [0]]  # Using High price as target
        self.len = len(self.y)

    def __len__(self):
        return (self.len - self.seq_len)-1

    def __getitem__(self, idx):
        x = np.transpose(self.X[idx:idx+self.seq_len])
        label = self.y[idx+self.seq_len+1]
        return torch.tensor(x, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    def normalize(self, X):
        X = np.transpose(X)
        X_norm = []
        for x in X:
            x = (x-np.min(x)) / (np.max(x)-np.min(x))
            X_norm.append(x)
        return np.transpose(X_norm)
    
class IstanbulStock(Dataset):
    def __init__(self, dir, seq_len=40):
        self.seq_len = seq_len
        data = np.loadtxt(dir, delimiter=',', skiprows=1, dtype=None)
        self.X = self.normalize(data[:, [0,1,2,3,4,5,6,7]])
        self.y = data[:, [0]]
        self.num_labels = len(np.unique(self.y))
        self.len = len(self.y)

    def __len__(self):
        return (self.len - self.seq_len)-1

    def __getitem__(self, idx):
        x = np.transpose(self.X[idx:idx+self.seq_len])
        label = self.y[idx+self.seq_len+1]

        return torch.tensor(x, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    def normalize(self, X):
        X = np.transpose(X)
        X_norm = []
        for x in X:
            x = (x-np.min(x)) / (np.max(x)-np.min(x))
            X_norm.append(x)
        return np.transpose(X_norm)

class AirQuality(Dataset):
    def __init__(self, dir, seq_len=24):
        self.seq_len = seq_len
        data = np.loadtxt(dir, delimiter=',', skiprows=1, dtype=None)
        self.X = self.normalize(data[:, [0,1,2,3,4,5,6,7,8,9,10,11]])
        self.y = data[:, [4]]
        self.num_labels = len(np.unique(self.y))
        self.len = len(self.y)

    def __len__(self):
        return (self.len - self.seq_len)-1

    def __getitem__(self, idx):
        x = np.transpose(self.X[idx:idx+self.seq_len])
        label = self.y[idx+self.seq_len+1]

        return torch.tensor(x, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    def normalize(self, X):
        X = np.transpose(X)
        X_norm = []
        for x in X:
            x = (x-np.min(x)) / (np.max(x)-np.min(x))
            X_norm.append(x)
        return np.transpose(X_norm)

class Traffic(Dataset):
    def __init__(self, dir, seq_len=24):
        self.seq_len = seq_len
        data = np.loadtxt(dir, delimiter=',', skiprows=1, dtype=None)
        self.X = self.normalize(data[:, [0,1,2,3,4,5,6,7]])
        self.y = data[:, [7]]
        self.num_labels = len(np.unique(self.y))
        self.len = len(self.y)

    def __len__(self):
        return (self.len - self.seq_len)-1

    def __getitem__(self, idx):
        x = np.transpose(self.X[idx:idx+self.seq_len])
        label = self.y[idx+self.seq_len+1]

        return torch.tensor(x, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    def normalize(self, X):
        X = np.transpose(X)
        X_norm = []
        for x in X:
            x = (x-np.min(x)) / (np.max(x)-np.min(x))
            X_norm.append(x)
        return np.transpose(X_norm)

class AppliancesEnergy1(Dataset):
    def __init__(self, dir, seq_len=144):
        self.seq_len = seq_len
        data = np.loadtxt(dir, delimiter=',', skiprows=1, dtype=None)
        self.X = self.normalize(data[:, 0:26])
        self.y = data[:, [0]]
        self.num_labels = len(np.unique(self.y))
        self.len = len(self.y)

    def __len__(self):
        return (self.len - self.seq_len)-1

    def __getitem__(self, idx):
        x = np.transpose(self.X[idx:idx+self.seq_len])
        label = self.y[idx+self.seq_len+1]

        return torch.tensor(x, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    def normalize(self, X):
        X = np.transpose(X)
        X_norm = []
        for x in X:
            x = (x-np.min(x)) / (np.max(x)-np.min(x))
            X_norm.append(x)
        return np.transpose(X_norm)
    
class AppliancesEnergy2(Dataset):
    def __init__(self, dir, seq_len=144):
        self.seq_len = seq_len
        data = np.loadtxt(dir, delimiter=',', skiprows=1, dtype=None)
        self.X = self.normalize(data[:, 0:26])
        self.y = data[:, [1]]
        self.num_labels = len(np.unique(self.y))
        self.len = len(self.y)

    def __len__(self):
        return (self.len - self.seq_len)-1

    def __getitem__(self, idx):
        x = np.transpose(self.X[idx:idx+self.seq_len])
        label = self.y[idx+self.seq_len+1]

        return torch.tensor(x, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    def normalize(self, X):
        X = np.transpose(X)
        X_norm = []
        for x in X:
            x = (x-np.min(x)) / (np.max(x)-np.min(x))
            X_norm.append(x)
        return np.transpose(X_norm)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, rel_emb):
        super(SelfAttention, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.heads = heads
        self.seq_len = seq_len
        self.module = module
        self.rel_emb = rel_emb
        modules = ['spatial', 'temporal', 'spatiotemporal', 'output']
        assert (modules.__contains__(module)), "Invalid module"

        if module == 'spatial' or module == 'temporal':
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

        #non-shared weights between heads for spatial and temporal modules
        if self.module == 'spatial' or self.module == 'temporal':
            values = self.values(x)
            keys = self.keys(x)
            queries = self.queries(x)
            values = values.reshape(N, self.seq_len, self.heads, self.embed_size)
            keys = keys.reshape(N, self.seq_len, self.heads, self.embed_size)
            queries = queries.reshape(N, self.seq_len, self.heads, self.embed_size)

        #shared weights between heads for spatio-temporal module
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
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len, device=self.device),
                    1)
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))

            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3) + S

        else:
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len, device=self.device),
                    1)
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))

            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3)

        #attention(N x Heads x Q_Len x K_len)
        #values(N x V_len x Heads x Head_dim)
        #z(N x Q_len x Heads*Head_dim)

        if self.module == 'spatial' or self.module == 'temporal':
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
        #pad a column of zeros on the left
        padded_qe = F.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        #take out first (padded) row
        return padded_qe[:,:,1:,:]

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, forward_expansion, rel_emb):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, seq_len, module, rel_emb=rel_emb)

        if module == 'spatial' or module == 'temporal':
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
    def __init__(self, seq_len, embed_size, num_layers, heads, device,
                 forward_expansion, module, output_size=1,
                 rel_emb=True):
        super(Encoder, self).__init__()
        self.module = module
        self.embed_size = embed_size
        self.device = device
        self.rel_emb = rel_emb
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.layers = nn.ModuleList(
            [
             TransformerBlock(embed_size, heads, seq_len, module, forward_expansion=forward_expansion, rel_emb = rel_emb)
             for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)

        out = self.fc_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, input_shape, output_size,
                 embed_size, num_layers, forward_expansion, heads):

        super(Transformer, self).__init__()
        self.batch_size, self.num_var, self.seq_len = input_shape
        self.device = device
        self.num_elements = self.seq_len*self.num_var
        self.embed_size = embed_size
        self.element_embedding = nn.Linear(self.seq_len, embed_size*self.seq_len)
        self.pos_embedding = nn.Embedding(self.seq_len, embed_size)
        self.variable_embedding = nn.Embedding(self.num_var, embed_size)

        self.temporal = Encoder(seq_len=self.seq_len,
                                embed_size=embed_size,
                                num_layers=num_layers,
                                heads=self.num_var,
                                device=self.device,
                                forward_expansion=forward_expansion,
                                module='temporal',
                                rel_emb=True)

        self.spatial = Encoder(seq_len=self.num_var,
                               embed_size=embed_size,
                               num_layers=num_layers,
                               heads=self.seq_len,
                               device=self.device,
                               forward_expansion=forward_expansion,
                               module = 'spatial',
                               rel_emb=True)

        self.spatiotemporal = Encoder(seq_len=self.seq_len*self.num_var,
                                      embed_size=embed_size,
                                      num_layers=num_layers,
                                      heads=heads,
                                      device=self.device,
                                      forward_expansion=forward_expansion,
                                      module = 'spatiotemporal',
                                      rel_emb=True)

        # consolidate embedding dimension
        self.fc_out1 = nn.Linear(embed_size, embed_size//2)
        self.fc_out2 = nn.Linear(embed_size//2, 1)

        #prediction
        self.out = nn.Linear((self.num_elements*3), output_size)

    def forward(self, x, dropout):
        batch_size = len(x)

        #process/embed input for temporal module
        positions = torch.arange(0, self.seq_len).expand(batch_size, self.num_var, self.seq_len).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_temporal = self.element_embedding(x).reshape(batch_size, self.num_elements, self.embed_size)
        x_temporal = F.dropout(self.pos_embedding(positions) + x_temporal, dropout)

        #process/embed input for spatial module
        x_spatial = torch.transpose(x, 1, 2).reshape(batch_size, self.num_var, self.seq_len)
        vars = torch.arange(0, self.num_var).expand(batch_size, self.seq_len, self.num_var).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_spatial = self.element_embedding(x_spatial).reshape(batch_size, self.num_elements, self.embed_size)
        x_spatial = F.dropout(self.variable_embedding(vars) + x_spatial, dropout)

        #process/embed input for spatio-temporal module
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

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def plot_metrics(train_metrics, val_metrics, metric_names, dataset):
    """Plot training and validation metrics and save to file"""
    plt.figure(figsize=(15, 5))
    fig, axes = plt.subplots(1, len(metric_names), figsize=(15, 5))
    
    for i, metric_name in enumerate(metric_names):
        axes[i].plot(train_metrics[metric_name], label=f'Train {metric_name}')
        axes[i].plot(val_metrics[metric_name], label=f'Val {metric_name}')
        axes[i].set_title(metric_name)
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{dataset}_metrics.png')
    plt.close()

def train_test(embed_size, heads, num_layers, dropout, forward_expansion, lr, batch_size, dir, dataset, NUM_EPOCHS=100, TEST_SPLIT=0.3):
    try:
        datasets = ['Uber', 'Traffic', 'AirQuality', 'AppliancesEnergy1', 'AppliancesEnergy2', 'IstanbulStock']
        assert (datasets.__contains__(dataset)), "Invalid dataset"

        #call dataset class
        if dataset == 'Uber':
            train_data = Uber(dir)
        elif dataset == 'Traffic':
            train_data = Traffic(dir)
        elif dataset == 'AirQuality':
            train_data = AirQuality(dir)
        elif dataset == 'AppliancesEnergy1':
            train_data = AppliancesEnergy1(dir)
        elif dataset == 'AppliancesEnergy2':
            train_data = AppliancesEnergy2(dir)
        elif dataset == 'IstanbulStock':
            train_data = IstanbulStock(dir)

        #split into train and test
        dataset_size = len(train_data)
        indices = list(range(dataset_size))
        split = int(np.floor(TEST_SPLIT * dataset_size))
        train_indices, test_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                sampler=train_sampler)
        test_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                    sampler=test_sampler)

        # Initialize metrics
        train_mse = MeanSquaredError().to(device)
        train_mae = MeanAbsoluteError().to(device)
        train_mape = MeanAbsolutePercentageError().to(device)
        
        val_mse = MeanSquaredError().to(device)
        val_mae = MeanAbsoluteError().to(device)
        val_mape = MeanAbsolutePercentageError().to(device)

        # Initialize model and optimizer
        inputs, _ = next(iter(train_dataloader))
        model = Transformer(inputs.shape, 1, embed_size=embed_size, num_layers=num_layers,
                           forward_expansion=forward_expansion, heads=heads).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        
        # Initialize tracking variables
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10 

        # Metric history
        history = {
            'train': {'MSE': [], 'MAE': [], 'MAPE': []},
            'val': {'MSE': [], 'MAE': [], 'MAPE': []}
        }

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=10, verbose=True)

        # Add gradient scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        
        # Training loop with mixed precision
        for epoch in range(NUM_EPOCHS):
            model.train()
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                # Use mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = model(inputs, dropout)
                    loss = loss_fn(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Clear cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Update metrics
                train_mse.update(outputs, labels)
                train_mae.update(outputs, labels)
                train_mape.update(outputs, labels)

            # Validation loop
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs, 0)
                    
                    # Update metrics
                    val_mse.update(outputs, labels)
                    val_mae.update(outputs, labels)
                    val_mape.update(outputs, labels)

            # Compute epoch metrics
            epoch_train_mse = train_mse.compute()
            epoch_train_mae = train_mae.compute()
            epoch_train_mape = train_mape.compute()
            
            epoch_val_mse = val_mse.compute()
            epoch_val_mae = val_mae.compute()
            epoch_val_mape = val_mape.compute()
            
            early_stopping(epoch_val_mse, model, f'best_model_{dataset}.pth')
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            if epoch_val_mse < best_val_loss:
                best_val_loss = epoch_val_mse
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'best_model_{dataset}.pth')
            else:
                patience_counter += 1

            # Store metrics
            history['train']['MSE'].append(epoch_train_mse.item())
            history['train']['MAE'].append(epoch_train_mae.item())
            history['train']['MAPE'].append(epoch_train_mape.item())
            
            history['val']['MSE'].append(epoch_val_mse.item())
            history['val']['MAE'].append(epoch_val_mae.item())
            history['val']['MAPE'].append(epoch_val_mape.item())

            # Reset metrics for next epoch
            train_mse.reset()
            train_mae.reset()
            train_mape.reset()
            val_mse.reset()
            val_mae.reset()
            val_mape.reset()

            # Early stopping check
            if epoch_val_mse < best_val_loss:
                best_val_loss = epoch_val_mse
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'best_model_{dataset}.pth')
            else:
                patience_counter += 1

            # Plot metrics every 5 epochs
            if epoch % 5 == 0:
                plot_metrics(
                    history['train'],
                    history['val'],
                    ['MSE', 'MAE', 'MAPE'],
                    dataset
                )
                print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
                print(f'Train MSE: {epoch_train_mse:.4f}, MAE: {epoch_train_mae:.4f}, MAPE: {epoch_train_mape:.4f}')
                print(f'Val MSE: {epoch_val_mse:.4f}, MAE: {epoch_val_mae:.4f}, MAPE: {epoch_val_mape:.4f}')

            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

        return history
    except Exception as e:
        print(f"Error in training: {e}")
        raise e
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Create necessary directories
os.makedirs('DATA', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Main execution block
if __name__ == "__main__":
    print("Starting STTRE training pipeline...")
    
    # Download Uber stock data first
    # print("Downloading Uber stock data...")
    # uber = yf.download('UBER', 
    #                   start='2019-05-10',
    #                   end='2024-01-01',
    #                   progress=False)
    # dir = 'DATA/uber_stock.csv'
    # uber.to_csv(dir)
    # print("Data downloaded successfully")
    
    # Create necessary directories
    os.makedirs('DATA', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # List of experiments to run
    experiments = [
        {
            'd': 32, 'h': 4, 'num_layers': 3, 'forward_expansion': 1,
            'dropout': 0.2, 'lr': 0.0001, 'batch_size': 256,
            'NUM_EPOCHS': 100, 'TEST_SPLIT': 0.3,
            'dir': 'DATA/uber_stock.csv', 'dataset': 'Uber'
        },
        # {
        #     'd': 8, 'h': 4, 'num_layers': 3, 'forward_expansion': 1,
        #     'dropout': 0.1, 'lr': 0.0001, 'batch_size': 32,
        #     'dir': '', 'dataset': 'AppliancesEnergy1'
        # },
    ]

    # Run all experiments
    for exp in experiments:
        print(f"\nStarting experiment with dataset: {exp['dataset']}")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory before each experiment
            history = train_test(**exp)
            print(f"Completed experiment with dataset: {exp['dataset']}")
        except Exception as e:
            print(f"Error in experiment with dataset {exp['dataset']}: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory after error
            continue

    print("\nAll experiments completed!")