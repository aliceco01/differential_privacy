"""
Configuration management for differential privacy autoencoder models.

This module provides configuration dataclasses to manage hyperparameters
and training settings, replacing global variables with structured configuration.
"""

from dataclasses import dataclass, field
from typing import Optional

try:
    import tensorflow as tf
except ImportError:
    tf = None


@dataclass
class AutoencoderConfig:
    """Configuration for basic autoencoder models.
    
    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        latent_size: Dimensionality of the latent space
        lambda_reg: L2 regularization strength
        noise_std: Standard deviation for differential privacy noise
        save_encodings: Whether to save encodings during training
        model_dir: Directory to save model checkpoints
        log_interval: Steps between logging
    """
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    latent_size: int = 2
    lambda_reg: float = 0.00001
    noise_std: float = 0.1
    save_encodings: bool = False
    model_dir: Optional[str] = None
    log_interval: int = 100


@dataclass
class VAEConfig(AutoencoderConfig):
    """Configuration for Variational Autoencoder models.
    
    Extends AutoencoderConfig with VAE-specific parameters.
    
    Attributes:
        beta: Weight for KL divergence term in VAE loss
        use_kl_annealing: Whether to use KL annealing during training
    """
    beta: float = 1.0
    use_kl_annealing: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training pipeline.
    
    Attributes:
        data_path: Path to training data
        validation_split: Fraction of data to use for validation
        shuffle_buffer: Size of shuffle buffer for dataset
        prefetch_buffer: Size of prefetch buffer for dataset
        checkpoint_dir: Directory to save checkpoints
        tensorboard_log_dir: Directory for TensorBoard logs
        save_freq: Frequency (in epochs) to save checkpoints
    """
    data_path: Optional[str] = None
    validation_split: float = 0.2
    shuffle_buffer: int = 10000
    prefetch_buffer: int = field(default_factory=lambda: tf.data.AUTOTUNE if tf else 1)
    checkpoint_dir: str = "./checkpoints"
    tensorboard_log_dir: str = "./logs"
    save_freq: int = 5


def get_default_config(model_type: str = "autoencoder") -> AutoencoderConfig:
    """Get default configuration for specified model type.
    
    Args:
        model_type: Type of model ("autoencoder" or "vae")
        
    Returns:
        Configuration object with default values
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type.lower() == "autoencoder":
        return AutoencoderConfig()
    elif model_type.lower() == "vae":
        return VAEConfig()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'autoencoder' or 'vae'.")
