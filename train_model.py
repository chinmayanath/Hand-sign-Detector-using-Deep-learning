"""
Sign Language Recognition Model Training Script

This module trains an LSTM-based neural network to recognize sign language gestures
from pre-processed landmark sequences. Features include:
- Configurable hyperparameters
- Comprehensive data validation
- Training metrics tracking and visualization
- Model versioning and checkpointing
- Detailed logging and error handling

Author: Refactored version
Date: 2025-10-20
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard,
    Callback
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    
    # Data configuration
    actions: List[str]
    data_path: Path
    model_path: Path
    sequence_length: int = 30
    feature_dim: int = 1662
    
    # Training hyperparameters
    test_size: float = 0.05
    validation_split: float = 0.15
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    random_state: int = 42
    
    # Model architecture
    lstm_units: List[int] = None
    dense_units: List[int] = None
    dropout_rates: List[float] = None
    
    # Callbacks configuration
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-7
    
    def __post_init__(self):
        """Set default values for list fields."""
        if self.lstm_units is None:
            self.lstm_units = [64, 128]
        if self.dense_units is None:
            self.dense_units = [64]
        if self.dropout_rates is None:
            self.dropout_rates = [0.2, 0.3, 0.2]
        
        # Convert string paths to Path objects
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
    
    def save(self, filepath: Path) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        # Convert Path objects to strings for JSON serialization
        config_dict['data_path'] = str(self.data_path)
        config_dict['model_path'] = str(self.model_path)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'ModelConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class TrainingMetrics(Callback):
    """Custom callback to track and log training metrics."""
    
    def __init__(self):
        super().__init__()
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        """Record metrics at the end of each epoch."""
        logs = logs or {}
        self.metrics['train_loss'].append(logs.get('loss', 0))
        self.metrics['train_accuracy'].append(logs.get('accuracy', 0))
        self.metrics['val_loss'].append(logs.get('val_loss', 0))
        self.metrics['val_accuracy'].append(logs.get('val_accuracy', 0))
        
        # Get current learning rate
        lr = float(self.model.optimizer.learning_rate.numpy())
        self.metrics['learning_rates'].append(lr)
        
        # Log progress
        logger.info(
            f"Epoch {epoch + 1}: "
            f"loss={logs.get('loss', 0):.4f}, "
            f"acc={logs.get('accuracy', 0):.4f}, "
            f"val_loss={logs.get('val_loss', 0):.4f}, "
            f"val_acc={logs.get('val_accuracy', 0):.4f}, "
            f"lr={lr:.2e}"
        )


class DatasetLoader:
    """Handles loading and validation of sign language datasets."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load sign language gesture sequences from the dataset directory.
        
        Returns:
            Tuple of (sequences, labels) as numpy arrays
            
        Raises:
            FileNotFoundError: If dataset directory doesn't exist
            ValueError: If no valid data files are found
        """
        if not self.config.data_path.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.config.data_path}"
            )
        
        sequences: List[np.ndarray] = []
        labels: List[int] = []
        stats = {action: 0 for action in self.config.actions}
        
        logger.info("="*60)
        logger.info("Loading dataset...")
        logger.info("="*60)
        
        for label_idx, action in enumerate(self.config.actions):
            action_path = self.config.data_path / action
            
            if not action_path.exists():
                logger.warning(f"Action directory not found: {action_path}")
                continue
            
            files = sorted(action_path.glob('*.npy'))
            logger.info(f"Processing action '{action}': {len(files)} files found")
            
            for file_path in files:
                try:
                    sequence = np.load(file_path)
                    
                    # Validate sequence shape
                    expected_shape = (self.config.sequence_length, self.config.feature_dim)
                    if sequence.shape != expected_shape:
                        logger.warning(
                            f"Skipping {file_path.name}: "
                            f"shape {sequence.shape} != expected {expected_shape}"
                        )
                        continue
                    
                    # Validate data quality (check for NaN/Inf)
                    if not self._validate_sequence(sequence, file_path):
                        continue
                    
                    sequences.append(sequence)
                    labels.append(label_idx)
                    stats[action] += 1
                    
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue
        
        if not sequences:
            raise ValueError("No valid data files found in dataset")
        
        # Log dataset statistics
        logger.info("\nDataset Statistics:")
        logger.info("-" * 40)
        total = sum(stats.values())
        for action, count in stats.items():
            percentage = (count / total * 100) if total > 0 else 0
            logger.info(f"  {action:12s}: {count:4d} samples ({percentage:5.1f}%)")
        logger.info("-" * 40)
        logger.info(f"  {'Total':12s}: {total:4d} samples")
        logger.info("="*60 + "\n")
        
        return np.array(sequences), np.array(labels)
    
    def _validate_sequence(self, sequence: np.ndarray, file_path: Path) -> bool:
        """
        Validate sequence data quality.
        
        Args:
            sequence: The sequence array to validate
            file_path: Path to the file (for logging)
            
        Returns:
            True if sequence is valid, False otherwise
        """
        if np.isnan(sequence).any():
            logger.warning(f"Skipping {file_path.name}: contains NaN values")
            return False
        
        if np.isinf(sequence).any():
            logger.warning(f"Skipping {file_path.name}: contains Inf values")
            return False
        
        return True


class ModelBuilder:
    """Handles model architecture construction."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def build(self) -> Sequential:
        """
        Build and compile the LSTM model architecture.
        
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential(name='SignLanguageRecognitionModel')
        num_classes = len(self.config.actions)
        
        # Build LSTM layers
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = (i < len(self.config.lstm_units) - 1)
            
            if i == 0:
                # First LSTM layer with input shape
                model.add(LSTM(
                    units,
                    return_sequences=return_sequences,
                    activation='relu',
                    input_shape=(self.config.sequence_length, self.config.feature_dim),
                    name=f'lstm_{i+1}'
                ))
            else:
                model.add(LSTM(
                    units,
                    return_sequences=return_sequences,
                    activation='relu',
                    name=f'lstm_{i+1}'
                ))
            
            # Add dropout after each LSTM layer
            if i < len(self.config.dropout_rates):
                model.add(Dropout(
                    self.config.dropout_rates[i],
                    name=f'dropout_lstm_{i+1}'
                ))
        
        # Build dense layers
        for i, units in enumerate(self.config.dense_units):
            model.add(Dense(
                units,
                activation='relu',
                name=f'dense_{i+1}'
            ))
            
            # Add dropout after dense layers
            dropout_idx = len(self.config.lstm_units) + i
            if dropout_idx < len(self.config.dropout_rates):
                model.add(Dropout(
                    self.config.dropout_rates[dropout_idx],
                    name=f'dropout_dense_{i+1}'
                ))
        
        # Output layer
        model.add(Dense(
            num_classes,
            activation='softmax',
            name='output'
        ))
        
        # Compile model
        optimizer = Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("\nModel Architecture:")
        logger.info("="*60)
        model.summary(print_fn=lambda x: logger.info(x))
        logger.info("="*60 + "\n")
        
        # Calculate and log total parameters
        total_params = model.count_params()
        logger.info(f"Total parameters: {total_params:,}\n")
        
        return model


class SignLanguageModelTrainer:
    """Main trainer class for sign language recognition models."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.model = None
        self.history = None
        self.metrics_tracker = TrainingMetrics()
        
        # Create output directories
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Create necessary directories for model storage."""
        self.config.model_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        (self.config.model_path / 'checkpoints').mkdir(exist_ok=True)
        (self.config.model_path / 'logs').mkdir(exist_ok=True)
        (self.config.model_path / 'plots').mkdir(exist_ok=True)
        
        logger.info(f"Model output directory: {self.config.model_path.absolute()}")
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training by encoding labels and splitting into sets.
        
        Args:
            X: Input sequences
            y: Labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for training...")
        
        # One-hot encode labels
        num_classes = len(self.config.actions)
        y_encoded = to_categorical(y, num_classes=num_classes).astype(np.float32)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Log data split information
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Train/Test ratio: {len(X_train)/len(X_test):.2f}\n")
        
        return X_train, X_test, y_train, y_test
    
    def _get_callbacks(self, run_name: str) -> List[Callback]:
        """
        Create training callbacks.
        
        Args:
            run_name: Unique identifier for this training run
            
        Returns:
            List of Keras callbacks
        """
        checkpoint_dir = self.config.model_path / 'checkpoints'
        log_dir = self.config.model_path / 'logs' / run_name
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=str(checkpoint_dir / f'{run_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=False
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_learning_rate,
                verbose=1,
                mode='min'
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                update_freq='epoch'
            ),
            
            # Custom metrics tracking
            self.metrics_tracker
        ]
        
        return callbacks
    
    def plot_training_history(self, run_name: str) -> None:
        """
        Plot and save training metrics.
        
        Args:
            run_name: Unique identifier for this training run
        """
        if not self.metrics_tracker.metrics['train_loss']:
            logger.warning("No metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16)
        
        epochs = range(1, len(self.metrics_tracker.metrics['train_loss']) + 1)
        
        # Plot loss
        axes[0, 0].plot(epochs, self.metrics_tracker.metrics['train_loss'], 
                       'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.metrics_tracker.metrics['val_loss'], 
                       'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[0, 1].plot(epochs, self.metrics_tracker.metrics['train_accuracy'], 
                       'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, self.metrics_tracker.metrics['val_accuracy'], 
                       'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot learning rate
        axes[1, 0].plot(epochs, self.metrics_tracker.metrics['learning_rates'], 
                       'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot accuracy difference
        acc_diff = np.array(self.metrics_tracker.metrics['train_accuracy']) - \
                   np.array(self.metrics_tracker.metrics['val_accuracy'])
        axes[1, 1].plot(epochs, acc_diff, 'm-', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_title('Overfitting Indicator (Train - Val Accuracy)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Difference')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.config.model_path / 'plots' / f'{run_name}_metrics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training plots saved to {plot_path}")
        plt.close()
    
    def save_training_summary(self, run_name: str, test_metrics: Dict[str, float]) -> None:
        """
        Save a summary of the training run.
        
        Args:
            run_name: Unique identifier for this training run
            test_metrics: Final test set metrics
        """
        summary = {
            'run_name': run_name,
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'final_metrics': test_metrics,
            'training_metrics': {
                'best_val_accuracy': max(self.metrics_tracker.metrics['val_accuracy']),
                'best_val_loss': min(self.metrics_tracker.metrics['val_loss']),
                'final_learning_rate': self.metrics_tracker.metrics['learning_rates'][-1],
                'epochs_trained': len(self.metrics_tracker.metrics['train_loss'])
            }
        }
        
        # Convert Path objects to strings for JSON
        summary['config']['data_path'] = str(self.config.data_path)
        summary['config']['model_path'] = str(self.config.model_path)
        
        summary_path = self.config.model_path / f'{run_name}_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to {summary_path}")
    
    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test set with detailed metrics.
        
        Args:
            X_test: Test sequences
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Dictionary of test metrics
        """
        logger.info("\n" + "="*60)
        logger.info("Evaluating model on test set...")
        logger.info("="*60)
        
        # Overall metrics
        test_loss, test_accuracy = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        # Per-class predictions
        predictions = self.model.predict(X_test, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Calculate per-class accuracy
        logger.info("\nPer-class Performance:")
        logger.info("-" * 40)
        for i, action in enumerate(self.config.actions):
            mask = true_classes == i
            if mask.sum() > 0:
                class_acc = (pred_classes[mask] == i).sum() / mask.sum()
                logger.info(f"  {action:12s}: {class_acc:.4f} ({mask.sum()} samples)")
        logger.info("-" * 40)
        
        logger.info(f"\nOverall Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Overall Test Loss: {test_loss:.4f}")
        logger.info("="*60 + "\n")
        
        return {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy)
        }
    
    def train(self) -> Dict[str, float]:
        """
        Execute the complete training pipeline.
        
        Returns:
            Dictionary containing final test metrics
            
        Raises:
            Exception: If training fails
        """
        run_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info("\n" + "="*60)
            logger.info("SIGN LANGUAGE MODEL TRAINING")
            logger.info("="*60 + "\n")
            
            # Save configuration
            self.config.save(self.config.model_path / f'{run_name}_config.json')
            
            # Load dataset
            loader = DatasetLoader(self.config)
            X, y = loader.load()
            
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(X, y)
            
            # Build model
            builder = ModelBuilder(self.config)
            self.model = builder.build()
            
            # Train model
            logger.info("Starting training...")
            logger.info("="*60 + "\n")
            
            self.history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=self._get_callbacks(run_name),
                verbose=2  # One line per epoch
            )
            
            logger.info("\n" + "="*60)
            logger.info("Training completed!")
            logger.info("="*60 + "\n")
            
            # Evaluate model
            test_metrics = self.evaluate_model(X_test, y_test)
            
            # Save final model
            final_model_path = self.config.model_path / f'{run_name}_final.h5'
            self.model.save(final_model_path)
            logger.info(f"Final model saved to {final_model_path}")
            
            # Generate plots
            self.plot_training_history(run_name)
            
            # Save training summary
            self.save_training_summary(run_name, test_metrics)
            
            logger.info("\n" + "="*60)
            logger.info("All outputs saved successfully!")
            logger.info(f"Model directory: {self.config.model_path.absolute()}")
            logger.info("="*60 + "\n")
            
            return test_metrics
            
        except KeyboardInterrupt:
            logger.warning("\nTraining interrupted by user (Ctrl+C)")
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point for the training script."""
    
    # Configuration
    config = ModelConfig(
        actions=["hello", "thanks", "iloveyou"],
        data_path=Path('dataset'),
        model_path=Path('models'),
        sequence_length=30,
        feature_dim=1662,
        test_size=0.05,
        epochs=30,
        batch_size=32,
        learning_rate=0.001,
        lstm_units=[64, 128],
        dense_units=[64],
        dropout_rates=[0.2, 0.3, 0.2]
    )
    
    # Initialize trainer
    trainer = SignLanguageModelTrainer(config)
    
    # Train model
    try:
        test_metrics = trainer.train()
        logger.info("Training completed successfully!")
        return 0
    except KeyboardInterrupt:
        logger.info("Training cancelled by user")
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())