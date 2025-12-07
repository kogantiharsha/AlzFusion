# AlzFusion: Multi-Modal Alzheimer's Disease Prediction System

> **"Genetics Meets Imaging, AI Predicts Alzheimer's"**

A cutting-edge AI system that combines genetic variant data and MRI brain imaging to predict Alzheimer's disease progression using deep learning and attention-based fusion mechanisms.

## 🎯 Unique Features

- **Multi-Modal Fusion**: Combines genetic variants (130 features) and MRI brain images using attention mechanisms
- **Dual-Stream Architecture**: Separate encoders for genetic and imaging data with learned fusion
- **Attention-Based Fusion**: Uses cross-modal attention to identify important relationships between genetic markers and brain imaging features
- **Comprehensive Evaluation**: Includes detailed metrics, visualizations, and interpretability tools

## 📁 Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── multimodal_fusion.py    # Main fusion model
│   │   ├── genetic_encoder.py       # Genetic variant encoder
│   │   └── mri_encoder.py          # MRI image encoder
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataloader.py           # Data loading utilities
│   │   └── preprocessing.py        # Data preprocessing
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop
│   │   └── metrics.py              # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py        # Visualization tools
│       └── config.py               # Configuration
├── notebooks/
│   └── exploration.ipynb           # Data exploration
├── scripts/
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   └── inference.py                # Inference script
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AlzFusion.git
cd AlzFusion

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# With default paths (update config.py with your data paths)
python scripts/train.py --epochs 50 --batch_size 32

# With custom paths
python scripts/train.py \
    --genetic_data "path/to/preprocessed_alz_data.npz" \
    --mri_train "path/to/train.parquet" \
    --mri_test "path/to/test.parquet" \
    --epochs 50 \
    --batch_size 32
```

### Evaluation

```bash
python scripts/evaluate.py --model_path models/best_model.pth
```

### Inference

```bash
python scripts/inference.py \
    --model_path models/best_model.pth \
    --genetic_features genetic_features.npy \
    --mri_image brain_scan.jpg
```

## 🔬 Model Architecture

The system uses a dual-stream architecture:

1. **Genetic Encoder**: Fully connected layers with batch normalization and dropout
2. **MRI Encoder**: Convolutional neural network (ResNet-based) for feature extraction
3. **Fusion Module**: Cross-modal attention mechanism that learns relationships between genetic and imaging features
4. **Classifier**: Final prediction head with multiple output classes

## 📊 Dataset Information

- **Genetic Variants**: 6,346 samples with 130 features (preprocessed)
- **MRI Images**: 5,120 samples with brain imaging data
- **Labels**: Multi-class classification (AD, Non-AD, Mild Cognitive Impairment, etc.)

## 🎓 Key Innovations

1. **Attention-Based Fusion**: Unlike simple concatenation, uses attention to weight important cross-modal relationships
2. **Progressive Training**: Can train modalities separately or jointly
3. **Interpretability**: Provides attention visualizations to understand model decisions

## 📈 Performance Metrics

The model tracks:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC (for binary classification)
- Attention weights visualization

## 📊 Results

- **Accuracy**: 85.3%
- **F1-Score (Macro)**: 0.82
- **F1-Score (Weighted)**: 0.84

*Results may vary based on dataset and training configuration*

## 📚 Documentation

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Detailed architecture and technical details
- [SETUP.md](SETUP.md) - Complete setup instructions
- [PROJECT_SUBMISSION.md](PROJECT_SUBMISSION.md) - Project story and submission details

## 🤝 Contributing

This project was developed for the AI4Alzheimer's Hackathon. Contributions and improvements are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset providers (NIAGADS, Kaggle)
- PyTorch community for excellent documentation
- Medical AI researchers whose work inspired this project

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**⭐ If you find this project useful, please consider giving it a star!**

