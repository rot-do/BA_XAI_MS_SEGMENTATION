import logging
import os
from datetime import datetime


def setup_logger(name: str = 'logger', log_dir: str = 'logs', level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a logger that logs messages to both a file and the console.

    Args:
        name (str): The name of the logger instance. Defaults to 'my_logger'.
        log_dir (str): Directory where log files will be stored. Will be created if it doesn't exist. Defaults to 'logs'.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.

    Returns:
        logging.Logger: A configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Log file with timestamp
    log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')
    log_path = os.path.join(log_dir, log_filename)

    # Formatter for logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
