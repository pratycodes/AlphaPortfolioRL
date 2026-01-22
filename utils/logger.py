import logging
import sys
import os
from datetime import datetime

def setup_logger(name: str = "quant_rl", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Configures a production-grade logger with Console and File handlers.
    
    Args:
        name (str): Name of the logger (usually __name__).
        log_dir (str): Directory to save log files.
        level (int): Logging threshold (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate logs if function is called multiple times
    if logger.hasHandlers():
        return logger

    # Create formatters
    # Detailed format: Time - Level - Module - Message
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. Console Handler (Stream to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # 2. File Handler
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Unique log file per run based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized. Writing logs to {log_file}")
    
    # Suppress chatty libraries
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    return logger