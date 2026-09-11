"""
Feature extraction module for Oil Degradation Classifier.
Implements HSV color features and GLCM texture features extraction.
"""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from typing import Dict, List
import logging

logger = logging.getLogger('OilDegradationClassifier')


class HSVFeatureExtractor:
    """
    Extracts color features from images using HSV color space.
    
    HSV (Hue-Saturation-Value) color space is more intuitive for color
    analysis than RGB, as it separates color information (hue) from
    intensity (value) and color purity (saturation).
    
    For each HSV channel, computes:
    - Statistical measures: mean, standard deviation, median
    - Histogram distribution: 10 bins per channel
    
    Total features: 3 channels × (3 stats + 10 histogram bins) = 39 features
    """
    
    def __init__(self, num_bins=10):
        """
        Initialize the HSV feature extractor.
        
        Args:
            num_bins: Number of histogram bins per channel
        """
        self.num_bins = num_bins
        logger.info(f"Initialized HSVFeatureExtractor with {num_bins} histogram bins")
    
    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract HSV color features from an image.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
        
        Returns:
            Dictionary containing HSV features with descriptive keys
        
        Raises:
            ValueError: If image is not in correct format
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
        
        # Convert RGB to HSV
        # OpenCV expects BGR, so we need to convert RGB -> BGR -> HSV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        
        # Split into H, S, V channels
        h_channel = image_hsv[:, :, 0]  # Hue: 0-179 in OpenCV
        s_channel = image_hsv[:, :, 1]  # Saturation: 0-255
        v_channel = image_hsv[:, :, 2]  # Value: 0-255
        
        features = {}
        
        # Extract features for each channel
        for channel_name, channel_data, max_val in [
            ('H', h_channel, 180),  # Hue range in OpenCV
            ('S', s_channel, 256),  # Saturation range
            ('V', v_channel, 256)   # Value range
        ]:
            # Statistical features
            features[f'{channel_name}_mean'] = float(np.mean(channel_data))
            features[f'{channel_name}_std'] = float(np.std(channel_data))
            features[f'{channel_name}_median'] = float(np.median(channel_data))
            
            # Histogram features
            hist, _ = np.histogram(channel_data, bins=self.num_bins, range=(0, max_val))
            # Normalize histogram to get probability distribution
            hist = hist.astype(float) / hist.sum()
            
            for i, bin_val in enumerate(hist):
                features[f'{channel_name}_hist_bin_{i}'] = float(bin_val)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get ordered list of feature names.
        
        Returns:
            List of feature names in the order they appear in extract()
        """
        feature_names = []
        
        for channel in ['H', 'S', 'V']:
            # Statistical features
            feature_names.extend([
                f'{channel}_mean',
                f'{channel}_std',
                f'{channel}_median'
            ])
            
            # Histogram features
            for i in range(self.num_bins):
                feature_names.append(f'{channel}_hist_bin_{i}')
        
        return feature_names


class GLCMFeatureExtractor:
    """
    Extracts texture features using Gray Level Co-occurrence Matrix (GLCM).
    
    GLCM analyzes spatial relationships between pixel intensities to
    characterize texture patterns. It computes how often pairs of pixels
    with specific values occur at specific spatial relationships.
    
    Features extracted:
    - Contrast: Measure of local intensity variation
    - Dissimilarity: Similar to contrast but with linear weighting
    - Homogeneity: Measure of closeness of distribution to diagonal
    - Energy (ASM): Measure of uniformity/orderliness
    - Correlation: Measure of linear dependency of gray levels
    
    Features are computed across multiple distances and angles, then averaged
    for rotation and scale invariance.
    """
    
    def __init__(self, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """
        Initialize the GLCM feature extractor.
        
        Args:
            distances: List of pixel pair distances to consider
            angles: List of angles (in radians) for pixel pair directions
        """
        self.distances = distances
        self.angles = angles
        logger.info(f"Initialized GLCMFeatureExtractor with distances={distances}, "
                   f"angles={[f'{a:.2f}' for a in angles]}")
    
    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract GLCM texture features from an image.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
        
        Returns:
            Dictionary containing GLCM texture features
        
        Raises:
            ValueError: If image is not in correct format
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
        
        # Convert to grayscale for GLCM computation
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Reduce gray levels to 256 for computational efficiency
        # GLCM computation is expensive with too many gray levels
        gray = gray.astype(np.uint8)
        
        # Compute GLCM
        # symmetric=True makes the matrix symmetric
        # normed=True normalizes the matrix
        glcm = graycomatrix(
            gray,
            distances=self.distances,
            angles=self.angles,
            levels=256,
            symmetric=True,
            normed=True
        )
        
        # Extract texture properties
        # Each property returns shape (distances, angles)
        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        energy = graycoprops(glcm, 'energy')
        correlation = graycoprops(glcm, 'correlation')
        
        # Average across all distances and angles for rotation invariance
        features = {
            'glcm_contrast': float(np.mean(contrast)),
            'glcm_dissimilarity': float(np.mean(dissimilarity)),
            'glcm_homogeneity': float(np.mean(homogeneity)),
            'glcm_energy': float(np.mean(energy)),
            'glcm_correlation': float(np.mean(correlation))
        }
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get ordered list of feature names.
        
        Returns:
            List of feature names in the order they appear in extract()
        """
        return [
            'glcm_contrast',
            'glcm_dissimilarity',
            'glcm_homogeneity',
            'glcm_energy',
            'glcm_correlation'
        ]


class FeatureExtractor:
    """
    Unified interface for extracting all features (HSV + GLCM).
    
    Combines HSV color features and GLCM texture features into a single
    feature vector for comprehensive image characterization.
    
    Total features: 39 (HSV) + 5 (GLCM) = 44 features per image
    """
    
    def __init__(self, config):
        """
        Initialize the unified feature extractor.
        
        Args:
            config: Configuration object with feature extraction parameters
        """
        self.config = config
        self.hsv_extractor = HSVFeatureExtractor(num_bins=10)
        self.glcm_extractor = GLCMFeatureExtractor(
            distances=config.GLCM_DISTANCES,
            angles=config.GLCM_ANGLES
        )
        logger.info("Initialized unified FeatureExtractor")
    
    def extract_from_image(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract all features from a single image.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
        
        Returns:
            Dictionary containing all features (HSV + GLCM)
        """
        # Extract HSV features
        hsv_features = self.hsv_extractor.extract(image)
        
        # Extract GLCM features
        glcm_features = self.glcm_extractor.extract(image)
        
        # Combine all features
        all_features = {**hsv_features, **glcm_features}
        
        return all_features
    
    def get_feature_names(self) -> List[str]:
        """
        Get ordered list of all feature names.
        
        Returns:
            List of all feature names (HSV + GLCM)
        """
        hsv_names = self.hsv_extractor.get_feature_names()
        glcm_names = self.glcm_extractor.get_feature_names()
        return hsv_names + glcm_names
    
    def extract_all_features(self, dataset, save_path=None):
        """
        Extract features from all images in a dataset.
        
        Args:
            dataset: PyTorch Dataset or iterable of images
            save_path: Optional path to save features as CSV
        
        Returns:
            pandas DataFrame with features and labels
        """
        import pandas as pd
        from tqdm import tqdm
        from PIL import Image
        
        logger.info(f"Extracting features from {len(dataset)} images...")
        
        all_features = []
        all_labels = []
        all_paths = []
        
        # Extract features from each image
        for idx in tqdm(range(len(dataset)), desc="Extracting features"):
            try:
                # Get image path and label from dataset
                img_path = dataset.image_paths[idx]
                label = dataset.labels[idx]
                
                # Load image as numpy array
                image = Image.open(img_path).convert('RGB')
                image_np = np.array(image)
                
                # Extract features
                features = self.extract_from_image(image_np)
                
                all_features.append(features)
                all_labels.append(label)
                all_paths.append(str(img_path))
                
            except Exception as e:
                logger.warning(f"Failed to extract features from {img_path}: {e}")
                continue
        
        # Create DataFrame
        features_df = pd.DataFrame(all_features)
        features_df['label'] = all_labels
        features_df['class_name'] = [dataset.class_names[l] for l in all_labels]
        features_df['image_path'] = all_paths
        
        logger.info(f"Extracted {len(features_df)} feature vectors")
        
        # Save to CSV if path provided
        if save_path:
            features_df.to_csv(save_path, index=False)
            logger.info(f"Saved features to {save_path}")
        
        return features_df
