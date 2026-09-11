"""
Utility functions for Oil Degradation Classifier.
Includes logging setup, device configuration, and helper functions.
"""

import logging
import random
import numpy as np
import torch
from pathlib import Path


def setup_logging(log_file=None):
    """
    Configure logging for the application.
    
    Args:
        log_file: Optional path to log file. If None, logs only to console.
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger('OilDegradationClassifier')
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the best available device (CUDA GPU or CPU).
    
    Returns:
        torch.device: Device to use for computation
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, train_losses, val_accuracies, 
                   best_val_acc, config, filepath):
    """
    Save model checkpoint with training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        epoch: Current epoch number
        train_losses: List of training losses
        val_accuracies: List of validation accuracies
        best_val_acc: Best validation accuracy achieved
        config: Configuration object
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'config': config
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load model checkpoint and restore training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        filepath: Path to checkpoint file
        device: Device to load model to
    
    Returns:
        tuple: (epoch, train_losses, val_accuracies, best_val_acc)
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return (
        checkpoint['epoch'],
        checkpoint['train_losses'],
        checkpoint['val_accuracies'],
        checkpoint['best_val_acc']
    )


def format_time(seconds):
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        str: Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
