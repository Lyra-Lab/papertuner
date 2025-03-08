import os
import logging

def setup_logging(verbose=False):
    """Configure logging based on verbosity level"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Define log file path
    log_file = os.path.join(log_dir, 'papertuner.log')

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all logs

    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler with appropriate verbosity
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if verbose else logging.WARNING)

    # Create file handler that logs everything
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)

    # Create formatters
    console_format = logging.Formatter('%(message)s')
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatters to handlers
    console.setFormatter(console_format)
    file_handler.setFormatter(file_format)

    # Add handlers to root logger
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)

    return log_file
