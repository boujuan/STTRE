import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from .data import BaseDataModule
from .model import STTRE

def train_sttre(config):
    # Create data module
    data_module = BaseDataModule(
        data_dir=config.data_dir,
        seq_len=config.seq_len,
        columns=config.columns,
        batch_size=config.batch_size
    )
    
    # Create model
    model = STTRE(
        input_shape=config.input_shape,
        output_size=config.output_size,
        embed_size=config.embed_size,
        num_layers=config.num_layers,
        forward_expansion=config.forward_expansion,
        heads=config.heads,
        learning_rate=config.learning_rate
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='sttre-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min'
    )
    
    # Logger
    logger = TensorBoardLogger("lightning_logs", name="sttre")
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=config.epochs,
        accelerator='auto',  # Automatically detect GPU/CPU
        devices='auto',
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
        gradient_clip_val=1.0,
        precision='16-mixed'  # Use mixed precision training
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module) 