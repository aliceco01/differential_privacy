# Quick Reference Card

## 🚀 Quick Start (30 seconds)
```bash
./setup.sh                    # Run setup
python train.py --epochs 5    # Train a model
```

## 📦 Installation
```bash
# Option 1: Quick setup
./setup.sh

# Option 2: Manual
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Option 3: Package install
pip install -e .
```

## 🎯 Common Commands

### Training
```bash
# Basic autoencoder
python train.py --model autoencoder --epochs 10

# VAE with custom params
python train.py --model vae --epochs 20 --learning-rate 0.0001

# Save encodings
python train.py --save-encodings

# Specify all parameters
python train.py --model autoencoder \
    --epochs 30 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --latent-size 10 \
    --noise-std 0.2 \
    --save-encodings
```

### Help
```bash
python train.py --help        # Show all options
cat README.md                 # Read docs
cat IMPROVEMENTS.md           # See improvement guide
cat SUMMARY.md                # See changes summary
```

## 📁 File Structure
```
differential_privacy/
├── README.md              # Main documentation
├── IMPROVEMENTS.md        # Improvement guide
├── SUMMARY.md            # Changes summary
├── requirements.txt       # Dependencies (UPDATED)
├── setup.py              # Package setup (NEW)
├── setup.sh              # Quick setup script (NEW)
├── .gitignore            # Git ignore rules (NEW)
│
├── config.py             # Configuration classes (NEW)
├── train.py              # CLI training script (NEW)
│
├── dp_ae.py              # DP Autoencoder (existing)
├── dp_pca.py             # Probabilistic PCA (existing)
├── mnist_cnn_tf2.py      # MNIST CNN (existing)
├── MNIST_autoencoder.py  # Keras AE (existing)
├── utils.py              # Utilities (existing)
│
└── presentation/         # Presentation files
```

## 🔧 Configuration Options

### Model Types
- `autoencoder` - Basic differential privacy autoencoder
- `vae` - Variational autoencoder

### Key Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--learning-rate` | 0.001 | Learning rate |
| `--latent-size` | 2 | Latent dimension |
| `--lambda-reg` | 0.00001 | L2 regularization |
| `--noise-std` | 0.1 | DP noise std dev |

### Paths
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-dir` | ./models | Model save directory |
| `--log-dir` | ./logs | Log directory |

### Flags
| Flag | Description |
|------|-------------|
| `--save-encodings` | Save encodings during training |
| `--no-gpu` | Disable GPU, use CPU only |

## 💻 Python API

### Using Config (New Way)
```python
from config import AutoencoderConfig
from dp_ae import basic_AE

# Create config
config = AutoencoderConfig(
    epochs=20,
    latent_size=5,
    learning_rate=0.001,
    noise_std=0.1
)

# Create model
model = basic_AE(save_encodings=True)
```

### Direct Usage (Old Way - Still Works)
```python
from dp_ae import basic_AE
import dp_ae

# Set globals
dp_ae.EPOCHS = 20
dp_ae.LATENT_SIZE = 5

# Create model
model = basic_AE()
```

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure you're in venv
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### GPU Issues
```bash
# Disable GPU
python train.py --no-gpu

# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python train.py
```

### Old TensorFlow Version
```bash
# Update dependencies
pip install --upgrade -r requirements.txt
```

## 📊 Output Files

After training, you'll find:
```
models/
├── checkpoint_epoch_5/     # Periodic checkpoints
├── checkpoint_epoch_10/
├── final_model/            # Final trained model
└── encodings/              # Saved encodings (if --save-encodings)
    ├── encodings.npy
    └── encodings_noisy_sd*.npy

logs/
└── training_*.log          # Training logs
```

## 🔍 What Changed?

### ✅ Fixed
- Broken dependencies (`pkg-resources`, old TensorFlow)
- Missing documentation (README, guides)
- No .gitignore
- Poor code organization

### ✨ Added
- `config.py` - Configuration management
- `train.py` - CLI interface
- `setup.py` - Package structure
- `setup.sh` - Quick setup
- Comprehensive documentation

### 📝 Updated
- `requirements.txt` - Modern dependencies

## 🎓 Learning Resources

### Documentation
1. `README.md` - Start here for overview
2. `IMPROVEMENTS.md` - Detailed improvement guide
3. `SUMMARY.md` - What changed and why

### Code
1. `config.py` - Configuration examples
2. `train.py` - Training pipeline
3. `dp_ae.py` - Model implementation

## 🚨 Important Notes

1. **Python Version**: Use Python 3.7-3.9 (TensorFlow compatibility)
2. **Virtual Environment**: Always use venv to avoid conflicts
3. **GPU Memory**: Enable memory growth (handled in train.py)
4. **Old Code**: Old API still works, but use new config when possible

## ⚡ Performance Tips

```bash
# Increase batch size for speed
python train.py --batch-size 128

# Use GPU
python train.py  # GPU auto-detected

# Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 python train.py

# Mixed precision (requires code update)
# TF_ENABLE_AUTO_MIXED_PRECISION=1 python train.py
```

## 🔗 Next Steps

1. ✅ Read README.md
2. ✅ Run setup.sh
3. ✅ Train test model: `python train.py --epochs 5`
4. ✅ Read IMPROVEMENTS.md for advanced usage
5. ✅ Customize for your needs

---

**Repository is now functional!** 🎉

For detailed information, see:
- `README.md` - Full documentation
- `IMPROVEMENTS.md` - Improvement guide
- `SUMMARY.md` - Complete changelog
