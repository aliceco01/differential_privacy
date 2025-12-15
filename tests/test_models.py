"""
Unit tests for autoencoder models.
"""

import pytest
import numpy as np
import tensorflow as tf
from dp_ae import basic_AE, basic_VAE


class TestBasicAE:
    """Test basic autoencoder model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample MNIST-like data."""
        # Create small batch of 28x28 images
        images = np.random.rand(4, 28, 28, 1).astype(np.float32)
        labels = np.array([0, 1, 2, 3])
        return images, labels
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = basic_AE(save_encodings=False)
        assert model is not None
        assert hasattr(model, 'encoding_layers')
        assert hasattr(model, 'decoding_layers')
        assert model.latent_ > 0
    
    def test_model_forward_pass(self, sample_data):
        """Test forward pass through model."""
        images, _ = sample_data
        model = basic_AE(save_encodings=False)
        
        # Forward pass
        output = model.call(images)
        
        # Check output shape matches input shape
        assert output.shape == images.shape
    
    def test_encoder_output_shape(self, sample_data):
        """Test encoder produces correct latent dimension."""
        images, _ = sample_data
        model = basic_AE(save_encodings=False)
        
        # Encode
        encoded = model.encode(images)
        
        # Check latent dimension
        assert encoded.shape[0] == images.shape[0]
        assert encoded.shape[1] == model.latent_
    
    def test_save_encodings_flag(self, sample_data):
        """Test that save_encodings flag works."""
        images, _ = sample_data
        model = basic_AE(save_encodings=True)
        
        # Process images
        _ = model.call(images)
        
        # Check encodings are saved
        assert len(model.encodings) > 0
    
    def test_loss_computation(self, sample_data):
        """Test loss computation."""
        images, _ = sample_data
        model = basic_AE(save_encodings=False)
        
        loss = model.mse_loss(images)
        
        # Loss should be a scalar tensor
        assert isinstance(loss, tf.Tensor)
        assert loss.shape == ()
        assert loss.numpy() >= 0


class TestBasicVAE:
    """Test variational autoencoder model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample MNIST-like data."""
        images = np.random.rand(4, 28, 28, 1).astype(np.float32)
        labels = np.array([0, 1, 2, 3])
        return images, labels
    
    def test_vae_initialization(self):
        """Test VAE can be initialized."""
        model = basic_VAE(save_encodings=False)
        assert model is not None
        assert hasattr(model, 'sampling')
    
    def test_vae_forward_pass(self, sample_data):
        """Test VAE forward pass."""
        images, _ = sample_data
        model = basic_VAE(save_encodings=False)
        
        output = model.call(images)
        assert output.shape == images.shape
    
    def test_vae_encoder_returns_distribution(self, sample_data):
        """Test VAE encoder returns mean, log_var, and sample."""
        images, _ = sample_data
        model = basic_VAE(save_encodings=False)
        
        z_mean, z_log_var, z = model.encode(images, return_vae=True)
        
        # Check all components are returned
        assert z_mean.shape[1] == model.latent_
        assert z_log_var.shape[1] == model.latent_
        assert z.shape[1] == model.latent_
    
    def test_vae_loss_computation(self, sample_data):
        """Test VAE loss (reconstruction + KL)."""
        images, _ = sample_data
        model = basic_VAE(save_encodings=False)
        
        loss = model.VAE_loss(images)
        
        # Loss should be a scalar
        assert isinstance(loss, tf.Tensor)
        assert loss.numpy() >= 0


class TestModelString:
    """Test model string representations."""
    
    def test_ae_string_representation(self):
        """Test autoencoder __str__ method."""
        model = basic_AE(save_encodings=False)
        model_str = str(model)
        assert 'basic_AE' in model_str
        assert 'z' in model_str  # latent size indicator


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
