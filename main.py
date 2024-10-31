from argparse import Namespace
from .train import train_sttre

if __name__ == "__main__":
    # Configuration
    config = Namespace(
        data_dir='~/STTRE_outputs/data/uber_stock.csv',
        seq_len=60,
        columns=[0, 1, 2, 3, 4],  # Adjust based on your data
        batch_size=32,
        input_shape=(32, 5, 60),  # (batch_size, num_variables, sequence_length)
        output_size=1,
        embed_size=64,
        num_layers=4,
        forward_expansion=2,
        heads=8,
        learning_rate=1e-3,
        epochs=100
    )
    
    # Train model
    train_sttre(config) 