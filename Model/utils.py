import sys
import torch

class Colors:
    # Regular colors
    BLUE = '\033[94m'      # Light/Bright Blue
    RED = '\033[91m'       # Light/Bright Red
    GREEN = '\033[92m'     # Light/Bright Green
    YELLOW = '\033[93m'    # Light/Bright Yellow
    CYAN = '\033[96m'      # Light/Bright Cyan
    MAGENTA = '\033[95m'   # Light/Bright Magenta
    
    # Bold colors
    BOLD_BLUE = '\033[1;34m'
    BOLD_GREEN = '\033[1;32m'
    BOLD_RED = '\033[1;31m'
    
    # Text style
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # End color
    ENDC = '\033[0m'
    
    # Emojis
    ROCKET = '🚀'
    HOURGLASS = '⌛'
    CHECK = '✅'
    CROSS = '❌'
    FIRE = '🔥'
    CHART = '📊'
    WARNING = '⚠️'
    BRAIN = '🧠'
    SAVE = '💾'
    STAR = '⭐'
    
    @classmethod
    def disable_colors(cls):
        """Disable colors if terminal doesn't support them"""
        for attr in dir(cls):
            if not attr.startswith('__') and isinstance(getattr(cls, attr), str):
                setattr(cls, attr, '')
    
    @staticmethod
    def supports_color():
        """Check if the terminal supports colors"""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

# Initialize colors based on terminal support
if not Colors.supports_color():
    Colors.disable_colors()

class DeviceManager:
    _device_info_printed = False

    @classmethod
    def get_device(cls):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if not cls._device_info_printed:
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")
                cls._device_info_printed = True
        else:
            device = torch.device('cpu')
            if not cls._device_info_printed:
                print("GPU not available, using CPU")
                cls._device_info_printed = True
        return device