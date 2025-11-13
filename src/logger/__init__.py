import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# -------------------------------
# Constants
# -------------------------------
LOG_DIR = "logs"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

# -------------------------------
# Create log folder if it doesn't exist
# -------------------------------
os.makedirs(LOG_DIR, exist_ok=True)

# -------------------------------
# Log file path with date
# -------------------------------
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y')}.log"
log_file_path = os.path.join(LOG_DIR, LOG_FILE)

# -------------------------------
# Create logger
# -------------------------------
def get_logger(name:str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture everything DEBUG and above

    # -------------------------------
    # File Handler (rotating)
    # -------------------------------
    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT
    )
    file_formatter = logging.Formatter(
        "LINE - %(lineno)d : [ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    # logger.addHandler(file_handler)

    # -------------------------------
    # Console Handler (Stream)
    # -------------------------------
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
         "LINE - %(lineno)d : [ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)  # Only INFO+ on console
    # logger.addHandler(console_handler)

    # -------------------------------
    # Optional: Avoid adding handlers multiple times
    # -------------------------------
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)


    return logger
