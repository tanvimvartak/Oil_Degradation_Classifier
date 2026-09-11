"""
Image preprocessing and augmentation module for Oil Degradation Classifier.
Provides transform pipelines for training and validation data.
"""

from torchvision import transforms
import logging

logger = logging.getLogger('OilDegradationClassifier')


class ImagePreprocessor:
    """
    Handles image preprocessing and augmentation pipelines.
    
    Provides separate transform pipelines for training (with augmentation)
    and validation (without augmentation) to ensure consistent preprocessing
    while preventing data leakage.
    
    Attributes:
        image_size: Target size for resizing images (height, width)
        normalize_mean: Mean values for normalization (RGB channels)
        normalize_std: Standard deviation values for normalization (RGB channels)
    """
    
    def __init__(self, image_size=(224, 224)):
        """
        Initialize the preprocessor.
        
        Args:
            image_size: Target size for resizing images as (height, width)
        """
        self.image_size = image_size
        
        # Use ImageNet normalization statistics as starting point
        # These work well for transfer learning and general image classification
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        
        logger.info(f"Initialized ImagePreprocessor with size {image_size}")
    
    def get_train_transforms(self) -> transforms.Compose:
        """
        Get the augmentation pipeline for training data.
        
        Applies data augmentation to increase dataset diversity and improve
        model generalization. Augmentations are moderate to preserve the
        essential characteristics of oil degradation.
        
        Augmentation steps:
        1. Resize to target size
        2. Random horizontal flip (50% probability)
        3. Random rotation (±15 degrees)
        4. Color jitter (brightness, contrast, saturation adjustments)
        5. Convert to tensor
        6. Normalize using ImageNet statistics
        
        Returns:
            Composed transform pipeline for training
        """
        train_transforms = transforms.Compose([
            # Resize to consistent dimensions
            transforms.Resize(self.image_size),
            
            # Random horizontal flip for geometric augmentation
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Random rotation to handle different camera angles
            transforms.RandomRotation(degrees=15),
            
            # Color jitter to simulate lighting variations
            # Adjust brightness, contrast, saturation, and hue
            transforms.ColorJitter(
                brightness=0.2,  # ±20% brightness
                contrast=0.2,    # ±20% contrast
                saturation=0.2,  # ±20% saturation
                hue=0.1          # ±10% hue
            ),
            
            # Random affine for slight perspective changes
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # ±10% translation
                scale=(0.9, 1.1)       # 90-110% scale
            ),
            
            # Convert PIL Image to tensor (scales to [0, 1])
            transforms.ToTensor(),
            
            # Normalize using ImageNet statistics
            transforms.Normalize(
                mean=self.normalize_mean,
                std=self.normalize_std
            )
        ])
        
        logger.info("Created training transforms with augmentation")
        return train_transforms
    
    def get_val_transforms(self) -> transforms.Compose:
        """
        Get the preprocessing pipeline for validation/test data.
        
        Applies only essential preprocessing without augmentation to ensure
        consistent and reproducible evaluation. No random operations are used.
        
        Preprocessing steps:
        1. Resize to target size
        2. Convert to tensor
        3. Normalize using ImageNet statistics
        
        Returns:
            Composed transform pipeline for validation
        """
        val_transforms = transforms.Compose([
            # Resize to consistent dimensions
            transforms.Resize(self.image_size),
            
            # Convert PIL Image to tensor (scales to [0, 1])
            transforms.ToTensor(),
            
            # Normalize using ImageNet statistics
            transforms.Normalize(
                mean=self.normalize_mean,
                std=self.normalize_std
            )
        ])
        
        logger.info("Created validation transforms without augmentation")
        return val_transforms
    
    def get_inference_transforms(self) -> transforms.Compose:
        """
        Get the preprocessing pipeline for inference on new images.
        
        Identical to validation transforms - no augmentation, only preprocessing.
        
        Returns:
            Composed transform pipeline for inference
        """
        return self.get_val_transforms()
    
    def denormalize(self, tensor):
        """
        Reverse normalization for visualization purposes.
        
        Converts normalized tensor back to [0, 1] range for display.
        
        Args:
            tensor: Normalized image tensor (C, H, W) or (B, C, H, W)
        
        Returns:
            Denormalized tensor in [0, 1] range
        """
        import torch
        
        # Handle both single image and batch
        if tensor.dim() == 3:
            # Single image (C, H, W)
            mean = torch.tensor(self.normalize_mean).view(3, 1, 1)
            std = torch.tensor(self.normalize_std).view(3, 1, 1)
        else:
            # Batch (B, C, H, W)
            mean = torch.tensor(self.normalize_mean).view(1, 3, 1, 1)
            std = torch.tensor(self.normalize_std).view(1, 3, 1, 1)
        
        # Denormalize: x = x * std + mean
        denormalized = tensor * std + mean
        
        # Clamp to [0, 1] range
        denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized


def get_transforms(config):
    """
    Convenience function to get both train and validation transforms.
    
    Args:
        config: Configuration object with IMAGE_SIZE parameter
    
    Returns:
        Tuple of (train_transforms, val_transforms)
    """
    preprocessor = ImagePreprocessor(image_size=config.IMAGE_SIZE)
    train_transforms = preprocessor.get_train_transforms()
    val_transforms = preprocessor.get_val_transforms()
    
    return train_transforms, val_transforms
