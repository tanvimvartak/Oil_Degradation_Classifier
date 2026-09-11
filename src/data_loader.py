"""
Data loading and management module for Oil Degradation Classifier.
Handles dataset loading, splitting, and batch preparation using PyTorch.
"""

import os
from pathlib import Path
from typing import Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger('OilDegradationClassifier')


class OilDegradationDataset(Dataset):
    """
    Custom PyTorch Dataset for oil degradation images.
    
    Loads images from directory structure where each subdirectory represents
    a degradation class: fresh.oil, 1.time.use.oil, 2.time.use.oil, 3.time.use.oil
    
    Attributes:
        root_dir: Root directory containing class subdirectories
        transform: Optional transform to be applied on images
        image_paths: List of paths to all images
        labels: List of corresponding integer labels
        class_names: List of class names in order
        class_to_idx: Dictionary mapping class names to indices
    """
    
    def __init__(self, root_dir: str, transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing class folders
            transform: Optional torchvision transforms to apply
        
        Raises:
            FileNotFoundError: If root_dir doesn't exist
            ValueError: If no valid images found
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Verify root directory exists
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")
        
        # Define class names in order (maps to labels 0, 1, 2, 3)
        self.class_names = [
            "fresh.oil",
            "1.time.use.oil",
            "2.time.use.oil",
            "3.time.use.oil"
        ]
        
        # Create class to index mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load all image paths and labels
        self._load_dataset()
        
        # Verify we found images
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in {root_dir}")
        
        logger.info(f"Loaded {len(self.image_paths)} images from {root_dir}")
    
    def _load_dataset(self):
        """
        Scan directories and build image path list with labels.
        Validates that each image file is readable.
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Get label for this class
            label = self.class_to_idx[class_name]
            
            # Scan for image files
            image_files = [
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in valid_extensions
            ]
            
            # Validate each image
            for img_path in image_files:
                try:
                    # Try to open image to verify it's valid
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                    
                except Exception as e:
                    logger.warning(f"Skipping invalid image {img_path}: {e}")
            
            logger.info(f"Found {len(image_files)} images in class '{class_name}'")
    
    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a single image and its label.
        
        Args:
            idx: Index of the image to load
        
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of images across classes.
        
        Returns:
            Dictionary mapping class names to image counts
        """
        distribution = {name: 0 for name in self.class_names}
        
        for label in self.labels:
            class_name = self.class_names[label]
            distribution[class_name] += 1
        
        return distribution


class DataManager:
    """
    Manages dataset loading, splitting, and DataLoader creation.
    
    Attributes:
        config: Configuration object with dataset parameters
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        class_names: List of class names
    """
    
    def __init__(self, config, train_transform=None, val_transform=None):
        """
        Initialize the data manager.
        
        Args:
            config: Configuration object
            train_transform: Transforms for training data
            val_transform: Transforms for validation data
        """
        self.config = config
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.train_loader = None
        self.val_loader = None
        self.class_names = None
    
    def load_dataset(self) -> Tuple[DataLoader, DataLoader, list]:
        """
        Load dataset and create train/validation DataLoaders.
        
        Returns:
            Tuple of (train_loader, val_loader, class_names)
        
        Raises:
            FileNotFoundError: If dataset directory doesn't exist
        """
        logger.info(f"Loading dataset from {self.config.DATA_DIR}")
        
        # Create full dataset
        full_dataset = OilDegradationDataset(
            root_dir=self.config.DATA_DIR,
            transform=None  # We'll apply transforms after splitting
        )
        
        self.class_names = full_dataset.class_names
        
        # Log class distribution
        distribution = full_dataset.get_class_distribution()
        logger.info("Class distribution:")
        for class_name, count in distribution.items():
            logger.info(f"  {class_name}: {count} images")
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(self.config.TRAIN_SPLIT * total_size)
        val_size = total_size - train_size
        
        logger.info(f"Splitting dataset: {train_size} train, {val_size} validation")
        
        # Split dataset
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.RANDOM_SEED)
        )
        
        # Apply transforms to splits
        if self.train_transform:
            train_dataset.dataset.transform = self.train_transform
        if self.val_transform:
            val_dataset.dataset.transform = self.val_transform
        
        # Create DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Created DataLoaders with batch size {self.config.BATCH_SIZE}")
        logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
        return self.train_loader, self.val_loader, self.class_names
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of images across classes.
        
        Returns:
            Dictionary mapping class names to image counts
        """
        if self.train_loader is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Access the original dataset through the subset
        original_dataset = self.train_loader.dataset.dataset
        return original_dataset.get_class_distribution()
