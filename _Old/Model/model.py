import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MeanSquaredError, MeanAbsoluteError

# Define all necessary model components

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, rel_emb):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.seq_len = seq_len
        self.module = module
        self.rel_emb = rel_emb
        modules = ['spatial', 'temporal', 'spatiotemporal', 'output']
        assert (module in modules), "Invalid module"

        if module in ['spatial', 'temporal']:
            self.head_dim = seq_len
            self.values = nn.Linear(self.embed_size, self.embed_size)
            self.keys = nn.Linear(self.embed_size, self.embed_size)
            self.queries = nn.Linear(self.embed_size, self.embed_size)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([self.heads, self.head_dim, self.embed_size]))
        else:
            self.head_dim = embed_size // heads
            assert (self.head_dim * heads == embed_size), "Embed size not divisible by heads"
            self.values = nn.Linear(self.head_dim, self.head_dim)
            self.keys = nn.Linear(self.head_dim, self.head_dim)
            self.queries = nn.Linear(self.head_dim, self.head_dim)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([1, self.seq_len, self.head_dim]))

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
            QE = torch.matmul(queries.transpose(1, 2), self.E.transpose(1, 2))
            QE = self._mask_positions(QE)
            S = self._skew(QE).contiguous().view(N, self.heads, self.seq_len, self.seq_len)
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len), 1).to(x.device)
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))
            attention = torch.softmax(qk / (self.embed_size ** 0.5), dim=3) + S
        else:
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len), 1).to(x.device)
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))
            attention = torch.softmax(qk / (self.embed_size ** 0.5), dim=3)

        if self.module in ['spatial', 'temporal']:
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len * self.heads, self.embed_size)
        else:
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len, self.heads * self.head_dim)

        z = self.fc_out(z)
        return z

    def _mask_positions(self, qe):
        L = qe.shape[-1]
        mask = torch.triu(torch.ones(L, L, device=self.device), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    def _skew(self, qe):
        padded_qe = F.pad(qe, [1, 0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        return padded_qe[:, :, 1:, :]

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, forward_expansion, rel_emb):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, seq_len, module, rel_emb=rel_emb)

        if module in ['spatial', 'temporal']:
            self.norm1 = nn.BatchNorm1d(seq_len * heads)
            self.norm2 = nn.BatchNorm1d(seq_len * heads)
        else:
            self.norm1 = nn.BatchNorm1d(seq_len)
            self.norm2 = nn.BatchNorm1d(seq_len)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.LeakyReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
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
            x = layer(x)
        out = self.fc_out(x)
        return out

class STTRE(nn.Module):
    def __init__(self, input_shape, output_size, embed_size, num_layers, forward_expansion, heads):
        super(STTRE, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.forward_expansion = forward_expansion
        self.heads = heads

        # Example layers (modify as per your architecture)
        self.attention = SelfAttention(embed_size, heads, seq_len=input_shape[1], module='spatial', rel_emb=True)
        self.layers = nn.ModuleList([
            nn.Linear(embed_size, embed_size) for _ in range(num_layers)
        ])
        self.output = nn.Linear(embed_size, output_size)

    def forward(self, x, dropout_rate=0.2):
        x = self.attention(x)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.output(x)
        return x

class STTREModel(pl.LightningModule):
    def __init__(self, input_shape, output_size, embed_size, num_layers, forward_expansion, heads, lr=1e-3, dropout=0.2):
        super(STTREModel, self).__init__()
        self.save_hyperparameters()
        self.model = STTRE(
            input_shape=input_shape,
            output_size=output_size,
            embed_size=embed_size,
            num_layers=num_layers,
            forward_expansion=forward_expansion,
            heads=heads
        )
        self.loss_fn = nn.MSELoss()
        self.metric_mse = MeanSquaredError()
        self.metric_mae = MeanAbsoluteError()

    def forward(self, x):
        batch_size, num_features, seq_len = x.shape
        x = x.view(batch_size * num_features, seq_len)  # Reshape to (batch_size * num_features, seq_len)
        x = self.model(x, dropout_rate=self.hparams.dropout)
        x = x.view(batch_size, num_features, -1)       # Reshape back if necessary
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.metric_mse(preds, y)
        self.metric_mae(preds, y)
        self.log('train_mse', self.metric_mse, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', self.metric_mae, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.metric_mse(preds, y)
        self.metric_mae(preds, y)
        self.log('val_mse', self.metric_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', self.metric_mae, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.metric_mse(preds, y)
        self.metric_mae(preds, y)
        self.log('test_mse', self.metric_mse, on_step=False, on_epoch=True)
        self.log('test_mae', self.metric_mae, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }