"""
Evaluation module for Oil Degradation Classifier.
Provides comprehensive model evaluation and reporting capabilities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger('OilDegradationClassifier')


class Evaluator:
    """
    Evaluates trained model and generates comprehensive reports.
    
    Provides functionality for:
    - Computing accuracy and per-class metrics
    - Generating confusion matrices
    - Visualizing predictions
    - Creating classification reports
    - Saving all results and visualizations
    
    Attributes:
        model: Trained PyTorch model
        val_loader: DataLoader for validation/test data
        class_names: List of class names
        device: Device to run evaluation on
    """
    
    def __init__(self, model, val_loader, class_names, device):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained PyTorch model
            val_loader: DataLoader for validation data
            class_names: List of class names in order
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.val_loader = val_loader
        self.class_names = class_names
        self.device = device
        
        logger.info("Initialized Evaluator")
    
    def evaluate(self):
        """
        Compute all evaluation metrics.
        
        Returns:
            Dictionary containing accuracy and per-class metrics
        """
        logger.info("Evaluating model...")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Evaluating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Store results
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute overall accuracy
        accuracy = accuracy_score(all_labels, all_preds) * 100
        
        # Compute per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )
        
        results = {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        }
        
        logger.info(f"Overall Accuracy: {accuracy:.2f}%")
        
        return results
    
    def generate_confusion_matrix(self, labels, predictions):
        """
        Generate confusion matrix from predictions.
        
        Args:
            labels: True labels
            predictions: Predicted labels
        
        Returns:
            Confusion matrix as numpy array
        """
        cm = confusion_matrix(labels, predictions)
        return cm
    
    def plot_confusion_matrix(self, cm, save_path):
        """
        Visualize confusion matrix as heatmap.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {save_path}")
    
    def get_sample_predictions(self, num_samples=16):
        """
        Get sample predictions with images for visualization.
        
        Args:
            num_samples: Number of samples to retrieve
        
        Returns:
            List of tuples (image, true_label, pred_label, confidence)
        """
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probs, 1)
                
                # Collect samples
                for i in range(images.size(0)):
                    if len(samples) >= num_samples:
                        return samples
                    
                    samples.append({
                        'image': images[i].cpu(),
                        'true_label': labels[i].item(),
                        'pred_label': predicted[i].item(),
                        'confidence': confidences[i].item()
                    })
                
                if len(samples) >= num_samples:
                    break
        
        return samples
    
    def plot_sample_predictions(self, samples, save_path, preprocessor=None):
        """
        Visualize sample predictions in a grid.
        
        Args:
            samples: List of sample dictionaries
            save_path: Path to save the plot
            preprocessor: ImagePreprocessor for denormalization (optional)
        """
        num_samples = len(samples)
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for idx, sample in enumerate(samples):
            ax = axes[idx]
            
            # Get image and denormalize if preprocessor provided
            image = sample['image']
            if preprocessor:
                image = preprocessor.denormalize(image)
            
            # Convert to numpy and transpose to (H, W, C)
            image_np = image.permute(1, 2, 0).numpy()
            image_np = np.clip(image_np, 0, 1)
            
            # Display image
            ax.imshow(image_np)
            
            # Create title with prediction info
            true_class = self.class_names[sample['true_label']]
            pred_class = self.class_names[sample['pred_label']]
            confidence = sample['confidence']
            
            # Color code: green if correct, red if wrong
            color = 'green' if sample['true_label'] == sample['pred_label'] else 'red'
            
            title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2%}"
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved sample predictions to {save_path}")
    
    def plot_training_history(self, train_losses, val_accuracies, save_path):
        """
        Plot training loss and validation accuracy curves.
        
        Args:
            train_losses: List of training losses per epoch
            val_accuracies: List of validation accuracies per epoch
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot training loss
        ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot validation accuracy
        ax2.plot(epochs, val_accuracies, 'g-', linewidth=2, label='Validation Accuracy')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Validation Accuracy over Epochs', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save figure
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training history to {save_path}")
    
    def generate_classification_report(self, labels, predictions):
        """
        Generate detailed classification report.
        
        Args:
            labels: True labels
            predictions: Predicted labels
        
        Returns:
            Classification report as string
        """
        report = classification_report(
            labels,
            predictions,
            target_names=self.class_names,
            digits=4
        )
        return report
    
    def save_results(self, results, train_losses, val_accuracies, results_dir, preprocessor=None):
        """
        Save all evaluation results and visualizations.
        
        Args:
            results: Dictionary from evaluate() method
            train_losses: List of training losses
            val_accuracies: List of validation accuracies
            results_dir: Directory to save results
            preprocessor: ImagePreprocessor for denormalization (optional)
        """
        logger.info(f"Saving results to {results_dir}")
        
        # Create results directory
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Generate and save confusion matrix
        cm = self.generate_confusion_matrix(results['labels'], results['predictions'])
        self.plot_confusion_matrix(cm, results_path / 'confusion_matrix.png')
        
        # Get and save sample predictions
        samples = self.get_sample_predictions(num_samples=16)
        self.plot_sample_predictions(samples, results_path / 'sample_predictions.png', preprocessor)
        
        # Save training history
        self.plot_training_history(train_losses, val_accuracies, results_path / 'training_history.png')
        
        # Generate and save classification report
        report = self.generate_classification_report(results['labels'], results['predictions'])
        with open(results_path / 'classification_report.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Overall Accuracy: {results['accuracy']:.2f}%\n\n")
            f.write(report)
            f.write("\n" + "="*70 + "\n")
            f.write("Per-Class Metrics:\n")
            f.write("="*70 + "\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {results['precision'][i]:.4f}\n")
                f.write(f"  Recall: {results['recall'][i]:.4f}\n")
                f.write(f"  F1-Score: {results['f1_score'][i]:.4f}\n")
                f.write(f"  Support: {results['support'][i]}\n")
        
        logger.info(f"Saved classification report to {results_path / 'classification_report.txt'}")
        
        # Save confusion matrix as CSV
        np.savetxt(
            results_path / 'confusion_matrix.csv',
            cm,
            delimiter=',',
            fmt='%d',
            header=','.join(self.class_names),
            comments=''
        )
        
        logger.info("All results saved successfully")
