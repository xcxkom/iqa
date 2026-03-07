import os
import logging

from .timer import PROGRAM_START_TIME


def get_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    
    log_file = os.path.join(log_dir, f"{PROGRAM_START_TIME.strftime('%Y%m%d_%H%M%S')}.log")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    console_formatter = logging.Formatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    file_formatter = logging.Formatter(
        fmt="[%(asctime)s - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, log_file