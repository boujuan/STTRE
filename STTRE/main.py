import os
from argparse import Namespace
from STTRE import train_sttre

def main():
    data_dir = os.path.expanduser('~/STTRE_outputs/data')
    os.makedirs(data_dir, exist_ok=True)
    
    config = Namespace(
        data_dir=os.path.join(data_dir, 'uber_stock.csv'),
        seq_len=60,
        columns=[5, 3, 4, 2, 6, 0],  # [Open, High, Low, Close, Volume, Price]
        batch_size=256,
        input_shape=(256, 5, 60),  # (batch_size, num_variables, seq_len)
        output_size=1,
        embed_size=64,
        num_layers=4,
        forward_expansion=2,
        heads=8,
        learning_rate=0.00005,
        epochs=1000
    )
    
    train_sttre(config)

if __name__ == "__main__":
    main() 