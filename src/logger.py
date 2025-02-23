import sys
import logging
import colorlog

def setup_logger() -> logging.Logger:
    """Sets up a global colored logger."""
    # Direct log messages to stderr
    handler = colorlog.StreamHandler(sys.stderr)
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white'}
    ))

    logger = colorlog.getLogger("stock_prediction")
    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger

# Global logger instance
logger = setup_logger()
