"""
Training orchestration module for Oil Degradation Classifier.
Handles model training, validation, and checkpoint management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import logging
from pathlib import Path

logger = logging.getLogger('OilDegradationClassifier')


class Trainer:
    """
    Orchestrates model training and validation.
    
    Manages the complete training loop including:
    - Forward/backward passes
    - Loss computation and optimization
    - Validation after each epoch
    - Learning rate scheduling
    - Model checkpointing
    - Metrics tracking
    
    Attributes:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration object
        device: Device to train on (CPU/GPU)
        criterion: Loss function
        optimizer: Optimization algorithm
        scheduler: Learning rate scheduler
        train_losses: List of training losses per epoch
        val_accuracies: List of validation accuracies per epoch
        best_val_acc: Best validation accuracy achieved
    """
    
    def __init__(self, model, train_loader, val_loader, config, device):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Configuration object
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function: CrossEntropyLoss for multi-class classification
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer: Adam with configurable learning rate
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE
        )
        
        # Learning rate scheduler: Reduce on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize validation accuracy
            factor=0.5,  # Reduce LR by half
            patience=5,  # Wait 5 epochs before reducing
            min_lr=1e-6
        )
        
        # Tracking metrics
        self.train_losses = []
        self.val_accuracies = []
        self.val_losses = []
        self.best_val_acc = 0.0
        self.start_epoch = 0
        
        logger.info(f"Initialized Trainer with {config.LEARNING_RATE} learning rate")
        logger.info(f"Training on device: {device}")
    
    def train_epoch(self) -> float:
        """
        Train for a single epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        # Calculate average loss and accuracy
        avg_loss = running_loss / len(self.train_loader)
        train_acc = 100 * correct / total
        
        return avg_loss, train_acc
    
    def validate(self) -> tuple:
        """
        Validate the model on validation set.
        
        Returns:
            Tuple of (validation_accuracy, validation_loss)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            
            for images, labels in pbar:
                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        # Calculate metrics
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100 * correct / total
        
        return val_acc, val_loss
    
    def train(self):
        """
        Execute the complete training loop.
        
        Trains for the configured number of epochs, validates after each epoch,
        saves best model, and updates learning rate based on validation performance.
        """
        logger.info(f"Starting training for {self.config.NUM_EPOCHS} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config.NUM_EPOCHS):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_acc, val_loss = self.validate()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log progress
            logger.info(
                f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(
                    self.config.MODEL_SAVE_PATH,
                    epoch,
                    is_best=True
                )
                logger.info(f"✓ New best model saved with validation accuracy: {val_acc:.2f}%")
            
            # Update learning rate based on validation accuracy
            self.scheduler.step(val_acc)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Early stopping check
            if current_lr < 1e-6:
                logger.info("Learning rate too small. Stopping training.")
                break
        
        # Training complete
        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time/60:.2f} minutes")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def save_checkpoint(self, filepath, epoch, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'val_losses': self.val_losses,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            # Also save as best model
            best_path = str(filepath).replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filepath):
        """
        Load model checkpoint to resume training.
        
        Args:
            filepath: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.train_losses = checkpoint['train_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        self.val_losses = checkpoint['val_losses']
        self.best_val_acc = checkpoint['best_val_acc']
        
        logger.info(f"Resumed from epoch {self.start_epoch}")
        logger.info(f"Best validation accuracy so far: {self.best_val_acc:.2f}%")
