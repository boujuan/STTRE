from pathlib import Path
import os

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