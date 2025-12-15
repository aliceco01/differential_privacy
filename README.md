# Differential Privacy for Anomaly Detection

This repository implements differential privacy techniques for machine learning models, focusing on autoencoders and PCA for anomaly detection tasks.

## Overview

The project explores privacy-preserving machine learning through:
- **Differentially Private Autoencoders (DP-AE)**: Neural network-based dimensionality reduction with privacy guarantees
- **Probabilistic PCA (pPCA)**: Statistical approach to dimensionality reduction with privacy considerations
- **MNIST Experiments**: Demonstrations using the MNIST dataset



## Setup Instructions

### Prerequisites
- Python 3.7-3.9 (required for TensorFlow compatibility)
- GPU support (optional, for faster training)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd differential_privacy
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a Differential Privacy Autoencoder

```python
from dp_ae import basic_AE
import tensorflow as tf

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0

# Create and train model
model = basic_AE(save_encodings=True)
# Training code here...
```

### Using Probabilistic PCA

```python
from dp_pca import pPCA

# Initialize model
model = pPCA(latent_dim=50)

# Fit to data
model.fit(X_train)

# Transform data
encoded = model.transform(X_test)
```

## Key Features

### Differential Privacy Autoencoder (`dp_ae.py`)
- Convolutional autoencoder architecture
- Noise injection for differential privacy
- Configurable latent dimension
- L2 regularization support

### Probabilistic PCA (`dp_pca.py`)
- Closed-form solution for dimensionality reduction
- RMSE evaluation metrics
- Visualization utilities (mosaic function)
- Privacy-preserving data analysis

## Configuration

Key hyperparameters (in `dp_ae.py`):
- `EPOCHS`: Training epochs
- `LAMBDA`: L2 regularization strength
- `LATENT_SIZE`: Dimensionality of encoded representation
- `BATCH_SIZE`: Training batch size
- `LEARNING_RATE`: Optimizer learning rate
- `SD`: Standard deviation for differential privacy noise

## Recent Improvements (December 2025)

### Completed Enhancements
1. **Modern TensorFlow 2.13+**: Updated from TensorFlow 2.0.1 with all secure dependencies
2. **Unit Tests**: Full test suite with pytest, coverage reporting, and CI-ready structure
3. **Configuration File Support**: YAML/JSON config loading with validation and examples
4. **Enhanced CLI**: Advanced command-line interface with config file support and rich options
5. **Professional Logging**: Structured logging system with file/console output and error tracking
6. **Privacy Budget Tracking**: Complete epsilon/delta tracking with composition methods
7. **DP Mechanisms Library**: Gaussian, Laplace mechanisms and Privacy Accountant implementation

### New Features

#### Running Tests
```bash
# Run all tests with coverage
python run_tests.py

# Or use pytest directly
pytest tests/ -v --cov

# Run specific tests
pytest tests/test_privacy.py -v
```

#### Using Configuration Files
```bash
# Train with YAML config
python train.py --config examples/config_autoencoder.yaml

# Train with JSON config  
python train.py --config examples/config_vae.json

# Save configuration
python train.py --model vae --epochs 20 --save-config my_config.yaml
```

#### Privacy Budget Tracking
```python
from privacy import PrivacyAccountant, GaussianMechanism

# Create accountant with total budget
accountant = PrivacyAccountant(epsilon=2.0, delta=1e-5)

# Create DP mechanism
mech = accountant.create_gaussian_mechanism(
    epsilon=0.5, delta=1e-6, sensitivity=1.0, name="gradient_noise"
)

# Add noise to data
noisy_data = mech.add_noise(data)

# Check budget status
print(accountant.get_report())
```

#### Advanced Logging
```python
from logging_utils import setup_logging

# Setup logging
logger = setup_logging(
    log_dir='./logs',
    log_level='INFO',
    log_to_file=True
)

logger.info("Training started")
```

## Project Structure

```
differential_privacy/
├── README.md                  # This file
├── requirements.txt           # Updated dependencies (TensorFlow 2.13+)
├── setup.py                   # Package configuration
├── config.py                  # Configuration dataclasses with YAML/JSON support
├── train.py                   # Enhanced CLI training interface
├── privacy.py                 # DP mechanisms and budget tracking
├── logging_utils.py           # Logging configuration
│
├── dp_ae.py                   # DP Autoencoder models
├── dp_pca.py                  # Probabilistic PCA
├── utils.py                   # Utility functions
│
├── examples/                  # Example configurations
│   ├── config_autoencoder.yaml
│   └── config_vae.json
│
└── tests/                     # Comprehensive test suite
    ├── conftest.py
    ├── test_config.py
    ├── test_models.py
    ├── test_privacy.py
    └── test_utils.py
```


## Future Enhancements

- [ ] Full migration to configuration objects across all modules
- [ ] TensorBoard integration for training visualization
- [ ] Distributed training support
- [ ] Pre-trained model zoo
- [ ] Web UI for experiment management
- [ ] Additional DP optimization algorithms (DP-Adam, DP-FTRL)



