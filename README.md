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

## Known Issues & Limitations

1. **Outdated Dependencies**: The codebase uses TensorFlow 2.0.1 (from 2019). Consider updating to TensorFlow 2.x for security and performance improvements.
2. **Mixed APIs**: Some files use Keras while others use TensorFlow, which could be standardized.
3. **Global Variables**: Configuration via global variables should be refactored to use configuration files or classes.
4. **Limited Documentation**: Function docstrings and inline comments need expansion.

## Future Improvements

- [ ] Update to modern TensorFlow versions
- [ ] Add comprehensive unit tests
- [ ] Implement configuration file support (YAML/JSON)
- [ ] Add command-line interface (CLI)
- [ ] Improve error handling and logging
- [ ] Add more privacy budget tracking
- [ ] Implement additional DP mechanisms

## Research Context

This project appears to be research-focused on:
- Telemetry fusion for anomaly detection
- Privacy-preserving machine learning
- Dimensionality reduction techniques
- Differential privacy guarantees in neural networks

## Contributing

When contributing, please:
1. Follow PEP 8 style guidelines
2. Add docstrings to all functions/classes
3. Include unit tests for new features
4. Update this README with any new functionality

