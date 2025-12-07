# Setup Guide for AlzFusion

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AlzFusion.git
cd AlzFusion
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Data Setup

### Option 1: Use Your Own Data

Place your data files in the following structure:
```
Datasets/
├── ALZ_Variant Dataset/
│   └── preprocessed_alz_data.npz
└── MRI Dataset/
    ├── train.parquet
    └── test.parquet
```

### Option 2: Download Datasets

1. Download the genetic variant dataset from NIAGADS
2. Download the MRI dataset from Kaggle
3. Preprocess the data using the provided notebooks

## Quick Test

```bash
# Test import
python -c "from src.models import MultiModalFusionModel; print('Setup successful!')"

# Run example
python example_usage.py --mode both
```

## Troubleshooting

### CUDA Issues
If you don't have CUDA, the system will automatically use CPU (slower but works).

### Import Errors
Make sure you're in the project root directory and the virtual environment is activated.

### Memory Issues
Reduce batch size in training scripts: `--batch_size 8` or `--batch_size 16`

## Next Steps

- Read [README.md](README.md) for project overview
- Check [QUICKSTART.md](QUICKSTART.md) for quick start guide
- See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for architecture details

