"""
Logging configuration and utilities for differential privacy project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: str = 'INFO',
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_filename: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration for the project.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        log_filename: Custom log filename (generated if None)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('differential_privacy')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        if log_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'train_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_path / log_filename)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path / log_filename}")
    
    return logger


def log_config(logger: logging.Logger, config: dict) -> None:
    """Log configuration parameters.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary to log
    """
    logger.info("=" * 60)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)


def log_model_summary(logger: logging.Logger, model) -> None:
    """Log model summary information.
    
    Args:
        logger: Logger instance
        model: Model to summarize
    """
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Latent dimension: {model.latent_}")
    logger.info(f"Noise std: {model.sd_}")
    logger.info(f"Lambda (L2 reg): {model.lambda_}")
    logger.info("=" * 60)


def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    step: int,
    train_loss: float,
    test_loss: Optional[float] = None
) -> None:
    """Log training progress.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        total_epochs: Total number of epochs
        step: Current step
        train_loss: Training loss value
        test_loss: Test loss value (optional)
    """
    msg = f"Epoch {epoch}/{total_epochs}, Step {step}, Train Loss: {train_loss:.6f}"
    if test_loss is not None:
        msg += f", Test Loss: {test_loss:.6f}"
    logger.info(msg)


def log_error(logger: logging.Logger, error: Exception, context: str = "") -> None:
    """Log error with context.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context about the error
    """
    logger.error("=" * 60)
    logger.error(f"Error occurred: {context}")
    logger.error(f"Exception type: {type(error).__name__}")
    logger.error(f"Exception message: {str(error)}")
    logger.error("=" * 60)
    logger.exception("Full traceback:")


class LoggerContextManager:
    """Context manager for temporary logging level changes."""
    
    def __init__(self, logger: logging.Logger, level: str):
        """Initialize context manager.
        
        Args:
            logger: Logger to modify
            level: Temporary logging level
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = logger.level
    
    def __enter__(self):
        """Enter context - change logging level."""
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore logging level."""
        self.logger.setLevel(self.old_level)


def get_logger(name: str = 'differential_privacy') -> logging.Logger:
    """Get or create logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
