"""
CNN model architectures for Oil Degradation Classifier.
Implements custom CNN and optional ResNet transfer learning models.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger('OilDegradationClassifier')


class OilDegradationCNN(nn.Module):
    """
    Custom CNN architecture for oil degradation classification.
    
    Architecture follows VGG-style design with repeated conv-conv-pool blocks.
    Progressive channel expansion: 3 → 32 → 64 → 128 → 256
    
    Structure:
    - 4 convolutional blocks with batch normalization
    - Each block: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool
    - 3 fully connected layers with dropout for classification
    - Output: 4 classes (fresh, 1st use, 2nd use, 3rd use)
    
    Input: (B, 3, 224, 224)
    Output: (B, 4) logits
    """
    
    def __init__(self, num_classes=4, dropout_rate=0.5):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability for regularization
        """
        super(OilDegradationCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional Feature Extractor
        self.features = nn.Sequential(
            # Block 1: 3 → 32 channels, 224 → 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 → 112
            
            # Block 2: 32 → 64 channels, 112 → 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 → 56
            
            # Block 3: 64 → 128 channels, 56 → 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 → 28
            
            # Block 4: 128 → 256 channels, 28 → 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 → 14
        )
        
        # Classifier
        # After 4 pooling layers: 224 → 112 → 56 → 28 → 14
        # Feature map size: 256 × 14 × 14 = 50,176
        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            # First FC layer: 50,176 → 512
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Second FC layer: 512 → 128
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),  # Lower dropout for second layer
            
            # Output layer: 128 → num_classes
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized OilDegradationCNN with {num_classes} classes")
    
    def _initialize_weights(self):
        """
        Initialize model weights using He initialization for ReLU networks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
        
        Returns:
            Output logits of shape (B, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self, x):
        """
        Extract intermediate feature maps for visualization.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
        
        Returns:
            Feature maps from the last convolutional layer
        """
        return self.features(x)


class OilDegradationResNet(nn.Module):
    """
    Transfer learning approach using pretrained ResNet18.
    
    Uses ResNet18 pretrained on ImageNet as feature extractor,
    replacing only the final fully connected layer for 4-class
    oil degradation classification.
    
    This approach can achieve better performance with limited data
    by leveraging features learned from millions of ImageNet images.
    
    Input: (B, 3, 224, 224)
    Output: (B, 4) logits
    """
    
    def __init__(self, num_classes=4, pretrained=True, freeze_backbone=False):
        """
        Initialize ResNet-based model.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone weights (only train FC layer)
        """
        super(OilDegradationResNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Optionally freeze backbone layers
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
            logger.info("Froze ResNet backbone layers")
        
        # Replace final fully connected layer
        # ResNet18 has 512 features before FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        logger.info(f"Initialized OilDegradationResNet with {num_classes} classes "
                   f"(pretrained={pretrained}, freeze_backbone={freeze_backbone})")
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
        
        Returns:
            Output logits of shape (B, num_classes)
        """
        return self.resnet(x)
    
    def unfreeze_backbone(self):
        """
        Unfreeze backbone layers for fine-tuning.
        Call this after initial training with frozen backbone.
        """
        for param in self.resnet.parameters():
            param.requires_grad = True
        logger.info("Unfroze ResNet backbone layers for fine-tuning")


def create_model(model_type='custom', num_classes=4, **kwargs):
    """
    Factory function to create model instances.
    
    Args:
        model_type: Type of model ('custom' or 'resnet')
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        Initialized model instance
    
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == 'custom':
        model = OilDegradationCNN(num_classes=num_classes, **kwargs)
    elif model_type == 'resnet':
        model = OilDegradationResNet(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'custom' or 'resnet'")
    
    return model


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_summary(model, input_size=(1, 3, 224, 224)):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size for testing
    """
    total_params, trainable_params = count_parameters(model)
    
    print("\n" + "="*70)
    print(f"Model: {model.__class__.__name__}")
    print("="*70)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*70)
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Input shape: {tuple(dummy_input.shape)}")
        print(f"Output shape: {tuple(output.shape)}")
        print("="*70 + "\n")
    except Exception as e:
        print(f"Error in forward pass: {e}")
        print("="*70 + "\n")
