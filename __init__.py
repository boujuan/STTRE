"""
STTRE (Spatial-Temporal Transformer Encoder) Package
"""

from .data import BaseDataModule
from .dataset import TimeSeriesDataset
from .encoder import Encoder
from .model import STTRE
from .train import train_sttre

__version__ = '0.1.0'
__author__ = 'Juan Manuel Boullosa Novo'

# Define what should be available when someone does `from STTRE import *`
__all__ = [
    'BaseDataModule',
    'TimeSeriesDataset',
    'Encoder',
    'STTRE',
    'train_sttre',
] 