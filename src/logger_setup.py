import logging
from logging.handlers import RotatingFileHandler

from .paths import LOG_DIR

def setup_logger(log_filename="model_eval.log", log_dir="../logs"):
    '''
    Sets up a rotating log file for model evaluation logging.
    '''
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / log_filename

    handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=10)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[handler]
    )

    return logging.getLogger("ModelEvaluation")

# Global logger instance
logger = setup_logger()