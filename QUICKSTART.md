# Quick Start Guide

Get up and running with the Multi-Modal Alzheimer's Prediction System in minutes!

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Training

```bash
# Train the model with default settings
python scripts/train.py

# Or with custom parameters
python scripts/train.py --epochs 50 --batch_size 32 --lr 0.001
```

## Quick Evaluation

```bash
# Evaluate a trained model
python scripts/evaluate.py --model_path models/best_model.pth
```

## Quick Inference

```bash
# Run inference on new data
python scripts/inference.py \
    --model_path models/best_model.pth \
    --genetic_features genetic_features.npy \
    --mri_image brain_scan.jpg
```

## Example Usage Script

```bash
# Run the example script to see everything in action
python example_usage.py --mode both
```

## Data Requirements

Make sure you have:
1. **Genetic data**: `Datasets-20251207T091459Z-1-001/Datasets/ALZ_Variant Datset/preprocessed_alz_data.npz`
2. **MRI training data**: `Datasets-20251207T091459Z-1-001/Datasets/MRI Dataset/train.parquet`
3. **MRI test data**: `Datasets-20251207T091459Z-1-001/Datasets/MRI Dataset/test.parquet`

## Expected Output

After training, you'll find:
- `models/best_model.pth` - Best model checkpoint
- `logs/` - TensorBoard logs (view with `tensorboard --logdir logs`)
- `results/confusion_matrix.png` - Confusion matrix visualization

## Troubleshooting

### Out of Memory?
- Reduce `--batch_size` (try 16 or 8)
- Use CPU: Set `DEVICE = "cpu"` in `src/utils/config.py`

### Data Size Mismatch?
- The system automatically handles mismatched dataset sizes
- It will use the minimum available size for alignment

### Missing Dependencies?
```bash
pip install --upgrade -r requirements.txt
```

## Next Steps

1. **Explore the code**: Check out `src/models/` for architecture details
2. **Customize**: Modify `src/utils/config.py` for your needs
3. **Visualize**: Use TensorBoard to monitor training: `tensorboard --logdir logs`
4. **Extend**: Add new modalities or improve the fusion mechanism

## Need Help?

- See `README.md` for detailed documentation
- See `PROJECT_SUMMARY.md` for architecture details
- Check `example_usage.py` for code examples

Happy training! 🚀

