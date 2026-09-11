"""
Configuration module for Oil Degradation Classifier.
Contains all hyperparameters and system settings.
"""

import os
from pathlib import Path


class Config:
    """
    Central configuration class for the Oil Degradation Classifier.
    
    Attributes:
        DATA_DIR: Root directory containing oil degradation image folders
        IMAGE_SIZE: Target size for resizing images (height, width)
        BATCH_SIZE: Number of images per batch during training
        TRAIN_SPLIT: Ratio of data used for training (rest for validation)
        NUM_CLASSES: Number of degradation stages to classify
        LEARNING_RATE: Initial learning rate for optimizer
        NUM_EPOCHS: Maximum number of training epochs
        DROPOUT_RATE: Dropout probability for regularization
        GLCM_DISTANCES: Pixel distances for GLCM computation
        GLCM_ANGLES: Angles (in radians) for GLCM computation
        MODEL_SAVE_PATH: Path to save the best model checkpoint
        RESULTS_DIR: Directory to save evaluation results and visualizations
        RANDOM_SEED: Seed for reproducibility
        NUM_WORKERS: Number of subprocesses for data loading
    """
    
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
    
    # Feature extraction settings
    GLCM_DISTANCES = [1, 2, 3]
    GLCM_ANGLES = [0, 0.785398, 1.5708, 2.35619]  # 0, π/4, π/2, 3π/4 in radians
    
    # Output settings
    MODEL_SAVE_PATH = "models/best_model.pth"
    RESULTS_DIR = "results"
    
    # System settings
    RANDOM_SEED = 42
    NUM_WORKERS = 4
    
    # Class names mapping
    CLASS_NAMES = [
        "fresh.oil",
        "1.time.use.oil",
        "2.time.use.oil",
        "3.time.use.oil"
    ]
    
    @staticmethod
    def create_directories():
        """Create necessary directories if they don't exist."""
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("src", exist_ok=True)
    
    def __repr__(self):
        """String representation of configuration."""
        config_str = "Configuration Settings:\n"
        config_str += f"  Data Directory: {self.DATA_DIR}\n"
        config_str += f"  Image Size: {self.IMAGE_SIZE}\n"
        config_str += f"  Batch Size: {self.BATCH_SIZE}\n"
        config_str += f"  Train Split: {self.TRAIN_SPLIT}\n"
        config_str += f"  Learning Rate: {self.LEARNING_RATE}\n"
        config_str += f"  Epochs: {self.NUM_EPOCHS}\n"
        config_str += f"  Number of Classes: {self.NUM_CLASSES}\n"
        return config_str
