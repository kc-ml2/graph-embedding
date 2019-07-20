import os.path
from pathlib import Path
from datetime import datetime

# Folders to manage
PROJECT_PATH = Path(__file__).resolve().parent.as_posix()
DATA_PATH = os.path.join(PROJECT_PATH, "data")

LOG_PATH = os.path.join(PROJECT_PATH, "logs")
CURRENT_TIME = datetime.now().strftime("%Y%m%d")
LOG_CURRENT_PATH = os.path.join(LOG_PATH, CURRENT_TIME)

CSV_CURRENT_PATH = os.path.join(LOG_CURRENT_PATH, "csv")
PLB_PATH = os.path.join(LOG_CURRENT_PATH, "playbacks")

MODEL_SAVED_PATH = os.path.join(PROJECT_PATH, "model_saved")

# Logging levels
LOG_LEVELS = {
    'DEBUG': {'lvl': 10, 'color': 'cyan'},
    'INFO': {'lvl': 20, 'color': 'white'},
    'WARNING': {'lvl': 30, 'color': 'yellow'},
    'ERROR': {'lvl': 40, 'color': 'red'},
    'CRITICAL': {'lvl': 50, 'color': 'red'},
}