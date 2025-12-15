"""
Main training script for differential privacy autoencoders.

This script provides a cleaner interface for training DP-AE and DP-VAE models
with proper configuration management and logging.

Usage:
    python train.py --model autoencoder --epochs 10 --latent-size 2
    python train.py --model vae --epochs 20 --learning-rate 0.0001
"""

import argparse
import os
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from datetime import datetime

from dp_ae import basic_AE, basic_VAE
from config import (AutoencoderConfig, VAEConfig, get_default_config, 
                    load_config, save_config)
import utils
import logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train differential privacy autoencoder models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python train.py --model autoencoder --epochs 10
  
  # Load config from file
  python train.py --config examples/config_autoencoder.yaml
  
  # Train with custom parameters
  python train.py --model vae --epochs 30 --latent-size 20 --learning-rate 0.0001
  
  # Save current config to file
  python train.py --model autoencoder --save-config my_config.yaml --no-train
        """
    )
    
    # Model selection
    parser.add_argument(
        '--model', 
        type=str, 
        default='autoencoder',
        choices=['autoencoder', 'vae'],
        help='Model type to train'
    )
    
    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (YAML or JSON). Overrides other parameters.'
    )
    parser.add_argument(
        '--save-config',
        type=str,
        help='Save current configuration to file and optionally exit'
    )
    parser.add_argument(
        '--no-train',
        action='store_true',
        help='Don\'t train, only save configuration (use with --save-config)'
    )
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--latent-size', type=int, default=2, help='Latent space dimension')
    parser.add_argument('--lambda-reg', type=float, default=0.00001, help='L2 regularization')
    parser.add_argument('--noise-std', type=float, default=0.1, help='DP noise standard deviation')
    
    # Paths
    parser.add_argument('--model-dir', type=str, default='./models', help='Model save directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='TensorBoard log directory')
    
    # Flags
    parser.add_argument('--save-encodings', action='store_true', help='Save encodings during training')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU training')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()


def setup_gpu(use_gpu=True):
    """Configure GPU settings."""
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("GPU disabled, using CPU only")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth to avoid allocating all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Found {len(gpus)} GPU(s), memory growth enabled")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU found, using CPU")


def load_mnist_data(batch_size=32):
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    
    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
        .shuffle(10000)\
        .batch(batch_size, drop_remainder=True)\
        .prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))\
        .batch(batch_size, drop_remainder=True)\
        .prefetch(tf.data.AUTOTUNE)
    
    print(f"Loaded {len(x_train)} training samples, {len(x_test)} test samples")
    return train_ds, test_ds, (x_train, y_train), (x_test, y_test)


def create_model(args):
    """Create model based on configuration."""
    if args.model == 'autoencoder':
        config = AutoencoderConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            latent_size=args.latent_size,
            lambda_reg=args.lambda_reg,
            noise_std=args.noise_std,
            save_encodings=args.save_encodings
        )
        # Update global variables (temporary until dp_ae.py is refactored)
        import dp_ae
        dp_ae.EPOCHS = args.epochs
        dp_ae.BATCH_SIZE = args.batch_size
        dp_ae.LEARNING_RATE = args.learning_rate
        dp_ae.LATENT_SIZE = args.latent_size
        dp_ae.LAMBDA = args.lambda_reg
        dp_ae.SD = args.noise_std
        
        model = basic_AE(save_encodings=args.save_encodings)
        print(f"Created basic autoencoder: {model}")
        
    elif args.model == 'vae':
        config = VAEConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            latent_size=args.latent_size,
            lambda_reg=args.lambda_reg,
            noise_std=args.noise_std,
            save_encodings=args.save_encodings
        )
        # Update global variables
        import dp_ae
        dp_ae.EPOCHS = args.epochs
        dp_ae.BATCH_SIZE = args.batch_size
        dp_ae.LEARNING_RATE = args.learning_rate
        dp_ae.LATENT_SIZE = args.latent_size
        dp_ae.LAMBDA = args.lambda_reg
        dp_ae.SD = args.noise_std
        
        model = basic_VAE(save_encodings=args.save_encodings)
        print(f"Created VAE model: {model}")
        
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    return model, config


class SimpleLogger:
    """Simple logger for tracking training progress."""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def log(self, step, train_images, train_recon, test_images, test_recon):
        """Log training progress."""
        # Could be extended to save sample images, etc.
        pass
    
    def write(self, message):
        """Write message to log file."""
        with open(self.log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        print(message)


def train(model, train_ds, test_ds, config, args):
    """Training loop."""
    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Configuration: {config}")
    
    # Create directories
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir)
    logger = SimpleLogger(log_dir)
    
    # Training loop
    iter_counter = 0
    for epoch in range(config.epochs):
        logger.write(f"\n{'='*60}")
        logger.write(f"Epoch {epoch + 1}/{config.epochs}")
        logger.write(f"{'='*60}")
        
        iter_counter = model.train_epoch(
            train_ds, 
            test_ds, 
            logger, 
            iter_counter_start=iter_counter,
            epoch_counter=epoch
        )
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.epochs:
            checkpoint_path = model_dir / f"checkpoint_epoch_{epoch+1}"
            checkpoint_path.mkdir(exist_ok=True)
            model.save(str(checkpoint_path))
            logger.write(f"Saved checkpoint at epoch {epoch + 1}")
    
    # Save final model
    final_model_path = model_dir / "final_model"
    final_model_path.mkdir(exist_ok=True)
    model.save(str(final_model_path))
    logger.write(f"\nTraining complete! Final model saved to {final_model_path}")
    
    # Save encodings if requested
    if args.save_encodings and len(model.encodings) > 0:
        encodings_path = model_dir / "encodings"
        encodings_path.mkdir(exist_ok=True)
        model.save_encodings_npy(str(encodings_path))
        logger.write(f"Encodings saved to {encodings_path}")
    
    return model


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("Differential Privacy Autoencoder Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Latent dimension: {args.latent_size}")
    print(f"Noise std: {args.noise_std}")
    print("=" * 60)
    
    # Setup
    setup_gpu(use_gpu=not args.no_gpu)
    
    # Load data
    train_ds, test_ds, train_data, test_data = load_mnist_data(args.batch_size)
    
    # Create model
    model, config = create_model(args)
    
    # Train
    trained_model = train(model, train_ds, test_ds, config, args)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
