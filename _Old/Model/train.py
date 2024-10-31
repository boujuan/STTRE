import os
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from dataset import STTREDatamodule
from dataset import split_dataset
from model import STTREModel
from config import Config
from utils import Colors

torch.set_float32_matmul_precision('medium')  # to leverage H100 GPU Tensor Cores

def main():
    # Initialize configuration
    Config.create_directories()

    # Initialize DataModule
    data_module = STTREDatamodule(data_dir=Config.DATA_DIR, batch_size=64, seq_len=60)

    # Define model parameters and training parameters
    model_params = {
        'input_shape': (64, 60),  # (num_var, seq_len)
        'output_size': 1,
        'embed_size': 60,
        'num_layers': 3,
        'forward_expansion': 1,
        'heads': 4
    }

    train_params = {
        'lr': 1e-3,
        'dropout': 0.2
    }

    # Initialize model
    model = STTREModel(
        input_shape=model_params['input_shape'],
        output_size=model_params['output_size'],
        embed_size=model_params['embed_size'],
        num_layers=model_params['num_layers'],
        forward_expansion=model_params['forward_expansion'],
        heads=model_params['heads'],
        lr=train_params['lr'],
        dropout=train_params['dropout']  # Pass dropout directly
    )

    # Initialize logger
    logger = TensorBoardLogger(
        save_dir=Config.SAVE_DIR,
        name='tensorboard_logs',
        version='STTRE_Model'
    )

    # Initialize callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=Config.MODEL_DIR,
        filename='sttre-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Determine accelerator and devices based on availability
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = torch.cuda.device_count()  # Automatically use all available GPUs
    else:
        accelerator = 'cpu'
        devices = None  # PyTorch Lightning will default to CPU

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=1000,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=50,
        enable_progress_bar=True  # Enables the progress bar
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Test the model
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    data_file = os.path.join(Config.DATA_DIR, 'uber_stock.csv')
    split_dataset(data_file)
    main()
