from pathlib import Path

SOURCE_DIR = Path(__file__).absolute().parent
PARENT_DIR = SOURCE_DIR.parent
DATA_DIR = PARENT_DIR / "data"
LOG_DIR = PARENT_DIR / "logs"
MODEL_DIR = PARENT_DIR / "models"
TUNED_PARAMS_DIR = PARENT_DIR / "tuned_params"
ENCODER_DIR = MODEL_DIR / 'label_encoders'
PLOT_DIR = PARENT_DIR / 'sleep_wave' / 'plots'
REPORT_DIR = PARENT_DIR / 'sleep_wave' / 'reports' 
CONFIG_DIR = PARENT_DIR / 'sleep_wave' / 'config'

# Remove this line later once it's in:
CONFIG_DIR.mkdir(parents=True, exist_ok=True) 