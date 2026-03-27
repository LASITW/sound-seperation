import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import logging

from model import OpenUnmixLSTM, AudioProcessor
from dataset import MUSDB18Dataset


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Training framework for Open-Unmix audio source separation model.
    
    Implements:
    - MSE loss in magnitude domain
    - Adam optimizer with learning rate scheduling
    - Early stopping
    - Checkpoint saving
    - Validation monitoring
    """
    
    def __init__(
        self,
        model: OpenUnmixLSTM,
        train_dataset: MUSDB18Dataset,
        val_dataset: MUSDB18Dataset,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
        learning_rate: float = 0.001,
        weight_decay: float = 0.00001,
        batch_size: int = 4,
        num_workers: int = 0,
    ):
        """
        Args:
            model: OpenUnmixLSTM model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            device: torch.device (cuda or cpu)
            checkpoint_dir: Directory to save checkpoints
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            batch_size: Training batch size
            num_workers: DataLoader workers
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Move model to device
        self.model.to(device)
        
        # Set normalization statistics from training data
        self.model.set_normalization(
            train_dataset.mean,
            train_dataset.std
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda',
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda',
        )
        
        # Loss function: MSE in magnitude domain
        self.criterion = nn.MSELoss()
        
        # Optimizer: Adam with L2 regularization (weight decay)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Learning rate scheduler: decay on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.3,
            patience=80,
            min_lr=1e-7,
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience = 140
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch+1} [TRAIN]",
            leave=False
        )
        
        for mixture, target in pbar:
            # Move to device
            mixture = mixture.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            mask = self.model(mixture)
            
            # Compute loss: MSE between predicted mask * mixture and target
            # mask * mixture should approximate target
            loss = self.criterion(mask * mixture, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate on validation set.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch+1} [VAL]",
            leave=False
        )
        
        for mixture, target in pbar:
            mixture = mixture.to(self.device)
            target = target.to(self.device)
            
            mask = self.model(mixture)
            loss = self.criterion(mask * mixture, target)
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False, force_save: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            force_save: Force save even if not best (for periodic snapshots)
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
        }
        
        # Save periodic snapshot every epoch for comparison
        if force_save or is_best:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch:03d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"✓ Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"✓ Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        logger.info(f"✓ Loaded checkpoint from {checkpoint_path}")
    
    def fit(self, num_epochs: int = 1000):
        """
        Full training loop with early stopping.
        
        Args:
            num_epochs: Maximum number of epochs to train
        """
        logger.info("=" * 60)
        logger.info("Starting training...")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Batch size: {self.train_loader.batch_size}")
        logger.info(f"Training batches per epoch: {len(self.train_loader)}")
        logger.info(f"Validation batches per epoch: {len(self.val_loader)}")
        logger.info("=" * 60)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            self.scheduler.step(val_loss)
            
            # Logging
            logger.info(
                f"Epoch {epoch+1:4d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Early stopping and checkpoint saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
                logger.info(f"✓ New best validation loss: {val_loss:.6f}")
            else:
                self.patience_counter += 1
                
                # Save snapshot every 10 epochs for comparison
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(force_save=True)
                
                if self.patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping triggered after {self.patience} epochs "
                        f"without improvement"
                    )
                    break
        
        # Save final training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        logger.info("=" * 60)


def main():
    """Main training script."""
    # MUSDB18 dataset path - CHANGE THIS to where your MUSDB18 dataset is located
    # Use the attached folder containing both train/ and test/
    MUSDB_PATH = "/Users/noelsaji/Desktop/PROJECTS/sound seperation/musdb18"
    
    # Configuration
    CONFIG = {
        'subset': 'train',
        'target': 'vocals',
        'chunk_duration': 7.0,  # Changed to 7 seconds
        'samples_per_track': 64,
        'n_fft': 4096,
        'hop_length': 1024,
        'batch_size': 4,
        'num_workers': 0,
        'learning_rate': 0.001,
        'weight_decay': 0.00001,
        'num_epochs': 1000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'musdb_path': MUSDB_PATH,
    }
    
    logger.info(f"Configuration: {json.dumps(CONFIG, indent=2)}")
    
    # Device
    device = torch.device(CONFIG['device'])
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Loading training dataset...")
    train_dataset = MUSDB18Dataset(
        subset='train',
        target=CONFIG['target'],
        chunk_duration=CONFIG['chunk_duration'],
        samples_per_track=CONFIG['samples_per_track'],
        n_fft=CONFIG['n_fft'],
        hop_length=CONFIG['hop_length'],
        augment=True,
        download=False,
        musdb_path=CONFIG['musdb_path'],
    )
    
    logger.info("Loading validation dataset...")
    val_dataset = MUSDB18Dataset(
        subset='test',
        target=CONFIG['target'],
        chunk_duration=CONFIG['chunk_duration'],
        samples_per_track=8,  # Fewer samples for validation
        n_fft=CONFIG['n_fft'],
        hop_length=CONFIG['hop_length'],
        augment=False,
        download=False,
        musdb_path=CONFIG['musdb_path'],
    )
    
    # Create model
    model = OpenUnmixLSTM(
        input_size=1 + CONFIG['n_fft'] // 2,
        hidden_size=512,
        num_layers=3,
        num_channels=2,  # Stereo
        dropout=0.0,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        checkpoint_dir='./checkpoints',
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
    )
    
    # Train
    trainer.fit(num_epochs=CONFIG['num_epochs'])


if __name__ == "__main__":
    main()
