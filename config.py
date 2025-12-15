"""
Configuration management for differential privacy autoencoder models.

This module provides configuration dataclasses to manage hyperparameters
and training settings, replacing global variables with structured configuration.
Supports loading/saving configurations from YAML and JSON files.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Union, Dict, Any
from pathlib import Path
import json
import yaml

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


def load_config_from_yaml(yaml_path: Union[str, Path], model_type: str = "autoencoder") -> AutoencoderConfig:
    """Load configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        model_type: Type of model configuration to create
        
    Returns:
        Configuration object loaded from YAML
        
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If YAML is invalid or model_type is unknown
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if model_type.lower() == "autoencoder":
        return AutoencoderConfig(**config_dict)
    elif model_type.lower() == "vae":
        return VAEConfig(**config_dict)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_config_from_json(json_path: Union[str, Path], model_type: str = "autoencoder") -> AutoencoderConfig:
    """Load configuration from JSON file.
    
    Args:
        json_path: Path to JSON configuration file
        model_type: Type of model configuration to create
        
    Returns:
        Configuration object loaded from JSON
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON is invalid or model_type is unknown
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Config file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    
    if model_type.lower() == "autoencoder":
        return AutoencoderConfig(**config_dict)
    elif model_type.lower() == "vae":
        return VAEConfig(**config_dict)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_config_to_yaml(config: AutoencoderConfig, yaml_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        yaml_path: Path where YAML file will be saved
    """
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = asdict(config)
    with open(yaml_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def save_config_to_json(config: AutoencoderConfig, json_path: Union[str, Path], indent: int = 2) -> None:
    """Save configuration to JSON file.
    
    Args:
        config: Configuration object to save
        json_path: Path where JSON file will be saved
        indent: Number of spaces for JSON indentation
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = asdict(config)
    with open(json_path, 'w') as f:
        json.dump(config_dict, f, indent=indent)


def load_config(config_path: Union[str, Path], model_type: str = "autoencoder") -> AutoencoderConfig:
    """Load configuration from file (auto-detects format from extension).
    
    Args:
        config_path: Path to configuration file (.yaml, .yml, or .json)
        model_type: Type of model configuration to create
        
    Returns:
        Configuration object loaded from file
        
    Raises:
        ValueError: If file extension is not supported
    """
    config_path = Path(config_path)
    suffix = config_path.suffix.lower()
    
    if suffix in ['.yaml', '.yml']:
        return load_config_from_yaml(config_path, model_type)
    elif suffix == '.json':
        return load_config_from_json(config_path, model_type)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}. Use .yaml, .yml, or .json")


def save_config(config: AutoencoderConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to file (auto-detects format from extension).
    
    Args:
        config: Configuration object to save
        config_path: Path where configuration file will be saved (.yaml, .yml, or .json)
        
    Raises:
        ValueError: If file extension is not supported
    """
    config_path = Path(config_path)
    suffix = config_path.suffix.lower()
    
    if suffix in ['.yaml', '.yml']:
        save_config_to_yaml(config, config_path)
    elif suffix == '.json':
        save_config_to_json(config, config_path)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}. Use .yaml, .yml, or .json")
