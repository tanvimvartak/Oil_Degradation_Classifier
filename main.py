"""
Main pipeline for Oil Degradation Classifier.
Orchestrates the complete workflow from data loading to evaluation.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import Config
from utils import setup_logging, set_seed, get_device
from preprocessing import ImagePreprocessor
from data_loader import DataManager
from feature_extraction import FeatureExtractor
from model import create_model, print_model_summary
from trainer import Trainer
from evaluator import Evaluator


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Oil Degradation Classifier - Deep Learning Pipeline'
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='custom',
        choices=['custom', 'resnet'],
        help='Model architecture to use (default: custom)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    # Feature extraction
    parser.add_argument(
        '--extract-features',
        action='store_true',
        help='Extract and save HSV+GLCM features'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training (useful for evaluation only)'
    )
    
    # Checkpoint
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        help='Only evaluate a trained model'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint for evaluation'
    )
    
    return parser.parse_args()


def main():
    """
    Main execution function.
    Orchestrates the complete pipeline.
    """
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("="*70)
    logger.info("Oil Degradation Classifier - Starting Pipeline")
    logger.info("="*70)
    
    # Load configuration
    config = Config()
    
    # Override config with command-line arguments
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    
    # Print configuration
    logger.info(config)
    
    # Set random seed for reproducibility
    set_seed(config.RANDOM_SEED)
    logger.info(f"Set random seed to {config.RANDOM_SEED}")
    
    # Get device
    device = get_device()
    
    # Create necessary directories
    config.create_directories()
    
    # ========================================================================
    # STEP 1: Data Loading
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Loading Dataset")
    logger.info("="*70)
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(image_size=config.IMAGE_SIZE)
    train_transforms = preprocessor.get_train_transforms()
    val_transforms = preprocessor.get_val_transforms()
    
    # Load dataset
    data_manager = DataManager(
        config,
        train_transform=train_transforms,
        val_transform=val_transforms
    )
    
    train_loader, val_loader, class_names = data_manager.load_dataset()
    
    logger.info(f"Classes: {class_names}")
    
    # ========================================================================
    # STEP 2: Feature Extraction (Optional)
    # ========================================================================
    if args.extract_features:
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Extracting Features")
        logger.info("="*70)
        
        # Create feature extractor
        feature_extractor = FeatureExtractor(config)
        
        # Extract features from training dataset
        # Access the original dataset through the subset
        original_dataset = train_loader.dataset.dataset
        
        features_df = feature_extractor.extract_all_features(
            original_dataset,
            save_path=f"{config.RESULTS_DIR}/features.csv"
        )
        
        logger.info(f"Extracted {len(features_df)} feature vectors")
        logger.info(f"Feature dimensions: {len(feature_extractor.get_feature_names())}")
    
    # ========================================================================
    # STEP 3: Model Creation
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Creating Model")
    logger.info("="*70)
    
    # Create model
    model = create_model(
        model_type=args.model,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    )
    
    # Print model summary
    print_model_summary(model, input_size=(1, 3, *config.IMAGE_SIZE))
    
    # Move model to device
    model = model.to(device)
    
    # ========================================================================
    # STEP 4: Training
    # ========================================================================
    if not args.skip_training and not args.evaluate_only:
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Training Model")
        logger.info("="*70)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # Train model
        trainer.train()
        
        # Get training history
        train_losses = trainer.train_losses
        val_accuracies = trainer.val_accuracies
        
    else:
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Skipping Training")
        logger.info("="*70)
        train_losses = []
        val_accuracies = []
    
    # ========================================================================
    # STEP 5: Evaluation
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Evaluating Model")
    logger.info("="*70)
    
    # Load checkpoint for evaluation if specified
    if args.checkpoint:
        logger.info(f"Loading model from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history if available
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            val_accuracies = checkpoint['val_accuracies']
    elif not args.skip_training and not args.evaluate_only:
        # Load best model from training
        logger.info(f"Loading best model from {config.MODEL_SAVE_PATH}")
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        val_loader=val_loader,
        class_names=class_names,
        device=device
    )
    
    # Evaluate model
    results = evaluator.evaluate()
    
    # Save all results
    evaluator.save_results(
        results=results,
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        results_dir=config.RESULTS_DIR,
        preprocessor=preprocessor
    )
    
    # ========================================================================
    # STEP 6: Summary
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETED")
    logger.info("="*70)
    logger.info(f"Final Validation Accuracy: {results['accuracy']:.2f}%")
    logger.info(f"Results saved to: {config.RESULTS_DIR}")
    logger.info(f"Model saved to: {config.MODEL_SAVE_PATH}")
    
    # Print per-class metrics
    logger.info("\nPer-Class Performance:")
    for i, class_name in enumerate(class_names):
        logger.info(
            f"  {class_name}: "
            f"Precision={results['precision'][i]:.4f}, "
            f"Recall={results['recall'][i]:.4f}, "
            f"F1={results['f1_score'][i]:.4f}"
        )
    
    logger.info("\n" + "="*70)
    logger.info("Thank you for using Oil Degradation Classifier!")
    logger.info("="*70)


if __name__ == '__main__':
    import torch  # Import here to avoid issues with argument parsing
    main()
