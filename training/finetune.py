import logging
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader

from mmaudio.model.networks import get_my_mmaudio
from mmaudio.model.flow_matching import FlowMatching
from training.memory_efficient_finetune import MemoryEfficientMMAudio, FineTuningOptimizer

log = logging.getLogger(__name__)

class FineTuningPipeline:
    """Complete fine-tuning pipeline for small datasets"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device('cuda')
        
        # Setup model
        self.setup_model()
        
        # Setup data augmentation
        self.setup_augmentation()
        
        # Metrics tracking
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def setup_model(self):
        """Load pretrained model with memory optimizations"""
        # Load base model
        base_model = get_my_mmaudio(self.cfg.model).to(self.device)
        
        # Load pretrained weights
        checkpoint = torch.load(
            self.cfg.finetune.pretrained_checkpoint, 
            map_location=self.device
        )
        base_model.load_state_dict(checkpoint, strict=True)
        log.info(f"Loaded pretrained weights from {self.cfg.finetune.pretrained_checkpoint}")
        
        # Wrap with memory-efficient version
        self.model = MemoryEfficientMMAudio(
            base_model,
            enable_checkpointing=self.cfg.gradient_checkpointing
        )
        
        # Initialize optimizer
        self.optimizer_wrapper = FineTuningOptimizer(
            self.model,
            lr=self.cfg.learning_rate,
            warmup_steps=self.cfg.warmup_steps
        )
        
        # Flow matching
        self.fm = FlowMatching(
            min_sigma=0,
            inference_mode='euler',
            num_steps=25  # Fast inference during training
        )
        
    def setup_augmentation(self):
        """Setup data augmentation for small datasets"""
        self.augmentations = nn.ModuleDict({
            'time_mask': TimeMasking(max_mask_len=20),
            'freq_mask': FrequencyMasking(max_mask_len=20),
            'mixup': MixUpAugmentation(alpha=self.cfg.augmentation.mixup_alpha)
        })
        
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Single training epoch with augmentation"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Apply augmentation
            if self.cfg.augmentation.time_masking:
                batch = self.augmentations['time_mask'](batch)
            if self.cfg.augmentation.frequency_masking:
                batch = self.augmentations['freq_mask'](batch)
                
            # Training step
            loss = self.optimizer_wrapper.training_step(
                batch, 
                accumulation_steps=self.cfg.gradient_accumulation_steps
            )
            
            total_loss += loss
            
            # Logging
            if batch_idx % 50 == 0:
                log.info(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss:.4f}")
                
            # Evaluation
            if (epoch * len(dataloader) + batch_idx) % self.cfg.eval_every == 0:
                val_loss = self.evaluate(val_dataloader)
                self.check_early_stopping(val_loss)
                
        return total_loss / len(dataloader)
        
    def evaluate(self, dataloader: DataLoader):
        """Evaluation with generation quality metrics"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Compute loss
                loss = self.model(batch)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(dataloader)
        log.info(f"Validation Loss: {avg_loss:.4f}")
        
        # Generate samples for quality check
        self.generate_samples(num_samples=4)
        
        return avg_loss
        
    def check_early_stopping(self, val_loss):
        """Early stopping logic"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            # Save best model
            self.save_checkpoint('best_model.pt')
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.cfg.early_stopping_patience:
            log.info("Early stopping triggered!")
            return True
            
        return False
        
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.base_model.state_dict(),
            'optimizer_state_dict': self.optimizer_wrapper.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.cfg
        }
        torch.save(checkpoint, Path(self.cfg.output_dir) / filename)
        log.info(f"Saved checkpoint: {filename}")


# Data Augmentation Classes
class TimeMasking(nn.Module):
    def __init__(self, max_mask_len=20):
        super().__init__()
        self.max_mask_len = max_mask_len
        
    def forward(self, batch):
        # Apply time masking to latent features
        if 'mean' in batch:
            B, T, D = batch['mean'].shape
            mask_len = torch.randint(1, self.max_mask_len, (1,)).item()
            mask_start = torch.randint(0, max(1, T - mask_len), (B,))
            
            for b in range(B):
                start = mask_start[b]
                batch['mean'][b, start:start+mask_len] = 0
                batch['std'][b, start:start+mask_len] = 1
                
        return batch


class FrequencyMasking(nn.Module):
    def __init__(self, max_mask_len=20):
        super().__init__()
        self.max_mask_len = max_mask_len
        
    def forward(self, batch):
        # Apply frequency masking to latent features
        if 'mean' in batch:
            B, T, D = batch['mean'].shape
            mask_len = torch.randint(1, min(self.max_mask_len, D//4), (1,)).item()
            mask_start = torch.randint(0, max(1, D - mask_len), (B,))
            
            for b in range(B):
                start = mask_start[b]
                batch['mean'][b, :, start:start+mask_len] = 0
                
        return batch


class MixUpAugmentation(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, batch):
        # MixUp augmentation for better generalization
        if self.training and torch.rand(1) < 0.5:
            B = batch['mean'].shape[0]
            
            # Sample lambda from Beta distribution
            lam = torch.tensor(
                np.random.beta(self.alpha, self.alpha, B)
            ).float().to(batch['mean'].device)
            
            # Random permutation for mixing
            index = torch.randperm(B)
            
            # Mix latent features
            lam = lam.view(-1, 1, 1)
            batch['mean'] = lam * batch['mean'] + (1 - lam) * batch['mean'][index]
            batch['std'] = lam * batch['std'] + (1 - lam) * batch['std'][index]
            
            # Mix visual features if present
            if 'clip_features' in batch:
                lam_visual = lam.view(-1, 1, 1)
                batch['clip_features'] = (lam_visual * batch['clip_features'] + 
                                        (1 - lam_visual) * batch['clip_features'][index])
                batch['sync_features'] = (lam_visual * batch['sync_features'] + 
                                        (1 - lam_visual) * batch['sync_features'][index])
                
        return batch


@hydra.main(version_base='1.3.2', config_path='../config', config_name='finetune_config.yaml')
def main(cfg: DictConfig):
    """Main fine-tuning entry point"""
    
    # Initialize pipeline
    pipeline = FineTuningPipeline(cfg)
    
    # Setup data loaders
    train_loader, val_loader = setup_finetuning_dataloaders(cfg)
    
    # Training loop
    for epoch in range(cfg.num_epochs):
        log.info(f"Starting epoch {epoch+1}/{cfg.num_epochs}")
        
        # Train
        train_loss = pipeline.train_epoch(train_loader, epoch)
        
        # Validate
        val_loss = pipeline.evaluate(val_loader)
        
        # Check early stopping
        if pipeline.check_early_stopping(val_loss):
            break
            
    log.info("Fine-tuning completed!")
    

if __name__ == "__main__":
    main()