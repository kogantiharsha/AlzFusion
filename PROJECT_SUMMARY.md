# Multi-Modal Alzheimer's Disease Prediction System - Project Summary

## 🎯 Project Overview

This project implements a **cutting-edge multi-modal deep learning system** that combines genetic variant data and MRI brain imaging to predict Alzheimer's disease progression. The system uses an innovative **attention-based fusion mechanism** to learn relationships between genetic markers and brain imaging features.

## ✨ Unique Features & Innovations

### 1. **Multi-Modal Fusion Architecture**
- **Dual-Stream Design**: Separate encoders for genetic variants (130 features) and MRI images
- **Attention-Based Fusion**: Cross-modal attention mechanism that learns which genetic markers are most relevant to specific brain imaging patterns
- **End-to-End Training**: Joint optimization of both modalities

### 2. **Genetic Variant Encoder**
- Fully connected neural network with batch normalization
- Processes 130 preprocessed genetic features
- Outputs 64-dimensional embeddings

### 3. **MRI Image Encoder**
- ResNet18-based convolutional neural network
- Pre-trained on ImageNet for transfer learning
- Extracts spatial features from brain images
- Outputs 64-dimensional embeddings

### 4. **Cross-Modal Attention Mechanism**
- **Bidirectional Attention**: Genetic features attend to MRI features and vice versa
- **Interpretability**: Attention weights reveal which genetic-imaging relationships are important
- **Adaptive Fusion**: Learns optimal combination of modalities

### 5. **Comprehensive Training Pipeline**
- **Progressive Training**: Can train modalities separately or jointly
- **Advanced Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Model Checkpointing**: Saves best model based on validation accuracy
- **TensorBoard Integration**: Real-time training visualization

## 📊 Dataset Information

### Genetic Variant Data
- **Source**: ALZ_Variant Dataset
- **Samples**: 6,346 total (5,076 train, 1,270 test)
- **Features**: 130 preprocessed features (binary encoded categorical + numerical)
- **Format**: Preprocessed .npz file with train/test splits

### MRI Brain Imaging Data
- **Source**: Kaggle MRI Alzheimer's Dataset
- **Samples**: 5,120 total
- **Format**: Parquet files with image bytes and labels
- **Labels**: Multi-class classification (0-8 representing different stages/conditions)

## 🏗️ Architecture Details

```
Input: Genetic Features (130) + MRI Images (224x224x3)
    ↓
┌─────────────────┐         ┌─────────────────┐
│ Genetic Encoder │         │   MRI Encoder   │
│  (FC Layers)    │         │   (ResNet18)    │
└────────┬────────┘         └────────┬────────┘
         │                            │
         └──────────┬─────────────────┘
                    ↓
         ┌──────────────────┐
         │ Cross-Modal      │
         │ Attention        │
         └────────┬─────────┘
                  ↓
         ┌──────────────────┐
         │  Fusion Layer    │
         │  (FC + Dropout)  │
         └────────┬─────────┘
                  ↓
         ┌──────────────────┐
         │  Classifier      │
         │  (9 Classes)     │
         └────────┬─────────┘
                  ↓
         Output: Class Probabilities
```

## 🚀 Key Advantages

1. **Better Performance**: Multi-modal fusion typically outperforms single-modal approaches
2. **Interpretability**: Attention weights show which genetic-imaging relationships matter
3. **Robustness**: Combining multiple data sources reduces overfitting
4. **Clinical Relevance**: Mirrors how clinicians use both genetic and imaging data
5. **Scalability**: Modular design allows easy addition of new modalities

## 📁 Project Structure

```
.
├── src/
│   ├── models/              # Model architectures
│   │   ├── genetic_encoder.py
│   │   ├── mri_encoder.py
│   │   └── multimodal_fusion.py
│   ├── data/                # Data loading
│   │   ├── dataloader.py
│   │   └── preprocessing.py
│   ├── training/            # Training utilities
│   │   ├── trainer.py
│   │   └── metrics.py
│   └── utils/               # Utilities
│       ├── config.py
│       └── visualization.py
├── scripts/                 # Executable scripts
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── notebooks/               # Jupyter notebooks
├── example_usage.py         # Quick start example
├── requirements.txt
└── README.md
```

## 🎓 Technical Highlights

### Model Specifications
- **Genetic Encoder**: 3 hidden layers (256→128→64) with batch norm and dropout
- **MRI Encoder**: ResNet18 backbone + projection head (512→256→64)
- **Fusion**: 128-dim fusion layer with dropout
- **Classifier**: 2-layer MLP (64→9 classes)
- **Total Parameters**: ~15-20M (depending on configuration)

### Training Details
- **Optimizer**: AdamW with weight decay
- **Loss Function**: CrossEntropyLoss
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduling
- **Batch Size**: 32 (configurable)
- **Data Augmentation**: Random horizontal flip, rotation for MRI images

### Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score (per-class and macro/weighted)
- Confusion Matrix
- Classification Report

## 🔬 Research & Clinical Impact

This system addresses a critical need in Alzheimer's research:
- **Early Detection**: Combining genetic risk factors with imaging biomarkers
- **Personalized Medicine**: Understanding individual genetic-imaging relationships
- **Research Tool**: Attention weights can reveal novel biological insights
- **Clinical Decision Support**: Assists clinicians in diagnosis and prognosis

## 🛠️ Usage Examples

### Training
```bash
python scripts/train.py --epochs 50 --batch_size 32
```

### Evaluation
```bash
python scripts/evaluate.py --model_path models/best_model.pth
```

### Inference
```bash
python scripts/inference.py --model_path models/best_model.pth \
    --genetic_features features.npy --mri_image brain_scan.jpg
```

## 📈 Future Enhancements

1. **Additional Modalities**: Add cognitive test scores, biomarkers, etc.
2. **Temporal Modeling**: Handle longitudinal data (multiple time points)
3. **Uncertainty Quantification**: Bayesian methods for confidence intervals
4. **Transfer Learning**: Pre-train on larger datasets
5. **Explainability**: SHAP values, Grad-CAM for MRI, feature importance

## 🏆 Competition/Hackathon Value

This project demonstrates:
- ✅ **Advanced ML Techniques**: Multi-modal fusion, attention mechanisms
- ✅ **Real-World Application**: Addresses actual medical challenge
- ✅ **Complete Pipeline**: Data loading → Training → Evaluation → Inference
- ✅ **Production-Ready Code**: Modular, documented, extensible
- ✅ **Innovation**: Unique attention-based fusion approach

## 📝 Notes

- The system handles mismatched dataset sizes gracefully
- Supports both labeled and unlabeled data
- Can be extended to binary classification or regression tasks
- Compatible with GPU and CPU training

---

**Developed for AI4Alzheimer's Hackathon**  
*Combining the power of genetics and imaging for better Alzheimer's prediction*

