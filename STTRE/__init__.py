"""
STTRE (Spatial-Temporal Transformer Encoder) Package

A Spatial-Temporal Transformer model for time series forecasting.
"""

from .data import BaseDataModule
from .dataset import TimeSeriesDataset
from .encoder import Encoder
from .model import STTRE
from .train import train_sttre

__version__ = '0.1.0'
__author__ = 'Juan Manuel Boullosa Novo'
__email__ = 'juanmanuel.boullosanovo@colorado.edu'
__license__ = 'MIT'
__copyright__ = '2024, Juan Manuel Boullosa Novo'
__description__ = 'Spatial-Temporal Transformer for Time Series Forecasting'
__url__ = 'https://github.com/boujuan/STTRE'

__all__ = [
    'BaseDataModule',
    'TimeSeriesDataset',
    'Encoder',
    'STTRE',
    'train_sttre',
]