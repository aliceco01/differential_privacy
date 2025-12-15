"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np
import tensorflow as tf


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    tf.random.set_seed(42)


@pytest.fixture
def mnist_sample_data():
    """Provide a small sample of MNIST-like data for testing."""
    num_samples = 10
    images = np.random.rand(num_samples, 28, 28, 1).astype(np.float32)
    labels = np.random.randint(0, 10, num_samples)
    return images, labels


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for model outputs."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return str(model_dir)


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary directory for logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return str(log_dir)
