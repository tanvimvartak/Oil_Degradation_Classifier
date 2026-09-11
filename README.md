# Oil Degradation Classifier

A deep learning image classification pipeline to assess edible oil degradation using custom CNN and traditional computer vision features (HSV color space and GLCM texture analysis).

## Overview

This project implements a comprehensive system for classifying sunflower oil images into four degradation stages:
- **fresh.oil**: Fresh, unused oil
- **1.time.use.oil**: Oil after first heating cycle
- **2.time.use.oil**: Oil after second heating cycle  
- **3.time.use.oil**: Oil after third heating cycle

The system combines:
- **Deep Learning**: Custom CNN architecture for end-to-end classification
- **Traditional Features**: HSV color features and GLCM texture features for interpretability
- **Transfer Learning**: Optional ResNet18 pretrained model

## Features

✅ Custom CNN with VGG-style architecture  
✅ HSV color space feature extraction (39 features)  
✅ GLCM texture feature extraction (5 features)  
✅ Data augmentation for limited datasets  
✅ Comprehensive evaluation metrics  
✅ Confusion matrix and sample prediction visualization  
✅ Training history plots  
✅ Model checkpointing and resumption  
✅ Command-line interface with flexible options  

## Project Structure

```
oil-degradation-classifier/
├── Datasets/                      # Dataset directory
│   ├── fresh.oil/                # Fresh oil images
│   ├── 1.time.use.oil/          # First use images
│   ├── 2.time.use.oil/          # Second use images
│   └── 3.time.use.oil/          # Third use images
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                # Configuration settings
│   ├── utils.py                 # Utility functions
│   ├── data_loader.py           # Dataset and DataLoader
│   ├── preprocessing.py         # Image preprocessing
│   ├── feature_extraction.py   # HSV and GLCM features
│   ├── model.py                 # CNN architectures
│   ├── trainer.py               # Training orchestration
│   └── evaluator.py             # Evaluation and visualization
├── models/                       # Saved model checkpoints
├── results/                      # Evaluation results
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   ├── training_history.png
│   ├── classification_report.txt
│   └── features.csv
├── tests/                        # Unit tests (optional)
├── main.py                       # Main pipeline script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd oil-degradation-classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Organize your dataset in the following structure:

```
Datasets/
├── fresh.oil/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 1.time.use.oil/
│   ├── image1.jpg
│   └── ...
├── 2.time.use.oil/
│   └── ...
└── 3.time.use.oil/
    └── ...
```

The system automatically:
- Assigns labels based on folder names
- Validates image files
- Splits data into train/validation sets (80/20 by default)

## Usage

### Basic Training

Train the model with default settings:

```bash
python main.py
```

### Advanced Options

```bash
# Use ResNet18 instead of custom CNN
python main.py --model resnet

# Custom hyperparameters
python main.py --epochs 100 --batch-size 64 --lr 0.0001

# Extract and save HSV+GLCM features
python main.py --extract-features

# Resume training from checkpoint
python main.py --resume models/best_model.pth

# Evaluate only (no training)
python main.py --evaluate-only --checkpoint models/best_model.pth
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model architecture (`custom` or `resnet`) | `custom` |
| `--epochs` | Number of training epochs | 50 |
| `--batch-size` | Batch size for training | 32 |
| `--lr` | Learning rate | 0.001 |
| `--extract-features` | Extract HSV+GLCM features to CSV | False |
| `--skip-training` | Skip training phase | False |
| `--resume` | Path to checkpoint to resume training | None |
| `--evaluate-only` | Only evaluate a trained model | False |
| `--checkpoint` | Path to model checkpoint for evaluation | None |

## Configuration

Edit `src/config.py` to customize:

```python
# Data settings
DATA_DIR = "Datasets"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8

# Model settings
NUM_CLASSES = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DROPOUT_RATE = 0.5

# Feature extraction
GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES = [0, π/4, π/2, 3π/4]
```

## Model Architecture

### Custom CNN

VGG-style architecture with 4 convolutional blocks:

```
Input (3, 224, 224)
    ↓
Block 1: Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool
    ↓
Block 2: Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool
    ↓
Block 3: Conv(128) → BN → ReLU → Conv(128) → BN → ReLU → MaxPool
    ↓
Block 4: Conv(256) → BN → ReLU → Conv(256) → BN → ReLU → MaxPool
    ↓
FC(512) → ReLU → Dropout(0.5)
    ↓
FC(128) → ReLU → Dropout(0.3)
    ↓
FC(4) → Output Logits
```

**Parameters**: ~13M trainable parameters

### ResNet18 (Transfer Learning)

Pretrained ResNet18 with modified final layer:
- Backbone: ResNet18 pretrained on ImageNet
- Final FC: 512 → 4 classes
- Option to freeze/unfreeze backbone

## Feature Extraction

### HSV Color Features (39 features)

For each channel (H, S, V):
- Mean, Standard Deviation, Median (3 stats)
- 10-bin histogram distribution (10 features)

**Total**: 3 channels × 13 features = 39 features

### GLCM Texture Features (5 features)

Computed across multiple distances and angles:
- Contrast: Local intensity variation
- Dissimilarity: Similar to contrast with linear weighting
- Homogeneity: Closeness to diagonal
- Energy: Uniformity/orderliness
- Correlation: Linear dependency of gray levels

## Data Augmentation

Training augmentations:
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine (translation, scale)
- Normalization (ImageNet statistics)

Validation: Only resize and normalization

## Evaluation Metrics

The system provides comprehensive evaluation:

1. **Overall Accuracy**: Percentage of correct predictions
2. **Per-Class Metrics**: Precision, Recall, F1-Score for each class
3. **Confusion Matrix**: Visual heatmap of predictions vs. true labels
4. **Sample Predictions**: Grid of images with predictions and confidence
5. **Training History**: Loss and accuracy curves over epochs

All results are saved to the `results/` directory.

## Results Interpretation

### Confusion Matrix
- Diagonal elements: Correct predictions
- Off-diagonal: Misclassifications
- Darker colors indicate higher counts

### Sample Predictions
- Green titles: Correct predictions
- Red titles: Incorrect predictions
- Confidence scores show model certainty

### Classification Report
- **Precision**: Of predicted class X, how many were actually X?
- **Recall**: Of actual class X, how many were predicted as X?
- **F1-Score**: Harmonic mean of precision and recall

## Training Tips

1. **Small Dataset**: Use data augmentation and consider transfer learning (ResNet)
2. **Overfitting**: Increase dropout rate or reduce model complexity
3. **Underfitting**: Increase model capacity or train longer
4. **Class Imbalance**: Check class distribution and consider weighted loss
5. **GPU Memory**: Reduce batch size if out of memory errors occur

## Troubleshooting

### Common Issues

**Issue**: `CUDA out of memory`  
**Solution**: Reduce batch size with `--batch-size 16`

**Issue**: `FileNotFoundError: Dataset directory not found`  
**Solution**: Ensure `Datasets/` folder exists with correct structure

**Issue**: Low accuracy  
**Solution**: 
- Check if images are correctly labeled
- Increase training epochs
- Try transfer learning with `--model resnet`
- Extract features to analyze class separability

**Issue**: Model not improving  
**Solution**:
- Check learning rate (try `--lr 0.0001` or `--lr 0.01`)
- Verify data augmentation isn't too aggressive
- Ensure sufficient training data per class

## Citation

If you use this code in your research, please cite:

```bibtex
@software{oil_degradation_classifier,
  title={Oil Degradation Classifier: Deep Learning for Edible Oil Quality Assessment},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/oil-degradation-classifier}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- scikit-image for GLCM implementation
- torchvision for pretrained models and transforms

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]

---

**Happy Classifying! 🛢️🔬**
