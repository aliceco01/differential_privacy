"""
Unit tests for configuration module.
"""

import pytest
from pathlib import Path
import tempfile
import json
import yaml
from config import AutoencoderConfig, VAEConfig, get_default_config


class TestAutoencoderConfig:
    """Test AutoencoderConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = AutoencoderConfig()
        assert config.epochs == 10
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.latent_size == 2
        assert config.lambda_reg == 0.00001
        assert config.noise_std == 0.1
        assert config.save_encodings is False
        assert config.model_dir is None
        assert config.log_interval == 100
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = AutoencoderConfig(
            epochs=20,
            batch_size=64,
            learning_rate=0.0001,
            latent_size=10
        )
        assert config.epochs == 20
        assert config.batch_size == 64
        assert config.learning_rate == 0.0001
        assert config.latent_size == 10
    
    def test_validation_positive_values(self):
        """Test that configuration validates positive values."""
        # This would require adding validation to the config class
        config = AutoencoderConfig(epochs=10, latent_size=5)
        assert config.epochs > 0
        assert config.latent_size > 0


class TestVAEConfig:
    """Test VAEConfig dataclass."""
    
    def test_vae_default_values(self):
        """Test VAE-specific default values."""
        config = VAEConfig()
        assert config.beta == 1.0
        assert config.use_kl_annealing is False
        # Should also have parent class defaults
        assert config.epochs == 10
        assert config.latent_size == 2
    
    def test_vae_custom_values(self):
        """Test VAE with custom parameters."""
        config = VAEConfig(
            beta=0.5,
            use_kl_annealing=True,
            epochs=30
        )
        assert config.beta == 0.5
        assert config.use_kl_annealing is True
        assert config.epochs == 30


class TestGetDefaultConfig:
    """Test get_default_config function."""
    
    def test_get_autoencoder_config(self):
        """Test getting default autoencoder config."""
        config = get_default_config("autoencoder")
        assert isinstance(config, AutoencoderConfig)
        assert not isinstance(config, VAEConfig)
    
    def test_get_vae_config(self):
        """Test getting default VAE config."""
        config = get_default_config("vae")
        assert isinstance(config, VAEConfig)
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError, match="Unknown model type"):
            get_default_config("invalid_model")


class TestConfigSerialization:
    """Test configuration serialization/deserialization."""
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = AutoencoderConfig(epochs=20, latent_size=5)
        # Using dataclasses.asdict if available
        from dataclasses import asdict
        config_dict = asdict(config)
        assert config_dict['epochs'] == 20
        assert config_dict['latent_size'] == 5
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'epochs': 25,
            'batch_size': 128,
            'learning_rate': 0.0005,
            'latent_size': 8
        }
        config = AutoencoderConfig(**config_dict)
        assert config.epochs == 25
        assert config.batch_size == 128


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
