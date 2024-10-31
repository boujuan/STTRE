import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder

class STTRE(L.LightningModule):
    def __init__(self, input_shape, output_size, embed_size, num_layers, 
                 forward_expansion, heads, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size, self.num_var, self.seq_len = input_shape
        self.num_elements = self.seq_len * self.num_var
        self.embed_size = embed_size
        self.learning_rate = learning_rate
        
        # Model layers (keeping original architecture)
        self.element_embedding = nn.Linear(self.seq_len, embed_size*self.seq_len)
        self.pos_embedding = nn.Embedding(self.seq_len, embed_size)
        self.variable_embedding = nn.Embedding(self.num_var, embed_size)
        
        # Encoders
        self.temporal = Encoder(seq_len=self.seq_len, embed_size=embed_size,
                              num_layers=num_layers, heads=self.num_var,
                              forward_expansion=forward_expansion,
                              module='temporal', rel_emb=True)
        
        self.spatial = Encoder(seq_len=self.num_var, embed_size=embed_size,
                             num_layers=num_layers, heads=self.seq_len,
                             forward_expansion=forward_expansion,
                             module='spatial', rel_emb=True)
        
        self.spatiotemporal = Encoder(seq_len=self.seq_len*self.num_var,
                                    embed_size=embed_size, num_layers=num_layers,
                                    heads=heads, forward_expansion=forward_expansion,
                                    module='spatiotemporal', rel_emb=True)
        
        # Output layers
        self.fc_out1 = nn.Linear(embed_size, embed_size//2)
        self.fc_out2 = nn.Linear(embed_size//2, 1)
        self.out = nn.Linear((self.num_elements*3), output_size)

    def forward(self, x):
        batch_size, num_var, seq_len = x.shape
        
        # Element embedding
        x_reshaped = x.reshape(batch_size * num_var, seq_len)
        element_emb = self.element_embedding(x_reshaped)
        element_emb = element_emb.reshape(batch_size, num_var, seq_len, self.embed_size)
        
        # Create position indices
        positions = torch.arange(0, seq_len).expand(batch_size, num_var, seq_len).to(x.device)
        variables = torch.arange(0, num_var).expand(batch_size, seq_len, num_var).transpose(1, 2).to(x.device)
        
        # Get embeddings
        pos_emb = self.pos_embedding(positions)
        var_emb = self.variable_embedding(variables)
        
        # Temporal attention
        temporal_in = element_emb + pos_emb
        temporal_out = self.temporal(temporal_in)
        
        # Spatial attention
        spatial_in = element_emb + var_emb
        spatial_in = spatial_in.transpose(1, 2)
        spatial_out = self.spatial(spatial_in).transpose(1, 2)
        
        # Spatiotemporal attention
        spatiotemporal_in = element_emb.reshape(batch_size, -1, self.embed_size)
        spatiotemporal_out = self.spatiotemporal(spatiotemporal_in)
        
        # Reshape outputs
        temporal_out = temporal_out.reshape(batch_size, -1)
        spatial_out = spatial_out.reshape(batch_size, -1)
        spatiotemporal_out = spatiotemporal_out.reshape(batch_size, -1)
        
        # Concatenate all outputs
        combined = torch.cat([temporal_out, spatial_out, spatiotemporal_out], dim=1)
        
        # Final output
        out = self.out(combined)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.mse_loss(y_hat, y)
        self.log('test_loss', test_loss)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        } 