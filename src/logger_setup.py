import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(log_filename="model_eval.log", log_dir="../logs"):
    """
    Sets up a rotating log file for model evaluation logging.
    """
    abs_log_dir = os.path.abspath(log_dir)
    os.makedirs(abs_log_dir, exist_ok=True)

    log_file = os.path.join(abs_log_dir, log_filename)

    handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=10)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[handler]
    )

    return logging.getLogger("ModelEvaluation")

# Global logger instance
logger = setup_logger()