"""
Memory-efficient wrapper for MMAudio fine-tuning.
Implements gradient checkpointing, layer freezing, and mixed precision optimizations.
"""

import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
import numpy as np

from mmaudio.model.networks import MMAudio

log = logging.getLogger(__name__)


class MemoryEfficientMMAudio(nn.Module):
    """
    Wrapper for MMAudio with memory optimizations for fine-tuning:
    - Gradient checkpointing
    - Layer freezing
    - Mixed precision support
    - Data augmentation
    """
    
    def __init__(
        self, 
        base_model: MMAudio,
        freeze_ratio: float = 0.5,
        enable_checkpointing: bool = True,
        dropout_rate: float = 0.1,
        enable_augmentation: bool = True
    ):
        super().__init__()
        self.base_model = base_model
        self.freeze_ratio = freeze_ratio
        self.enable_checkpointing = enable_checkpointing
        self.enable_augmentation = enable_augmentation
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Apply layer freezing
        self._freeze_layers()
        
        # Setup gradient checkpointing
        if enable_checkpointing:
            self._setup_checkpointing()
            
        # Initialize augmentation modules
        if enable_augmentation:
            self._setup_augmentation()
        
    def _freeze_layers(self):
        """Freeze early layers to save memory and prevent catastrophic forgetting"""
        
        # Always freeze input projections (they're well-trained)
        modules_to_freeze = [
            self.base_model.audio_input_proj,
            self.base_model.clip_input_proj,
            self.base_model.sync_input_proj,
            self.base_model.text_input_proj
        ]
        
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
                
        # Freeze early transformer blocks
        if hasattr(self.base_model, 'joint_blocks'):
            total_blocks = len(self.base_model.joint_blocks)
            freeze_until = int(total_blocks * self.freeze_ratio)
            
            for i in range(freeze_until):
                for param in self.base_model.joint_blocks[i].parameters():
                    param.requires_grad = False
                    
            log.info(f"Frozen {freeze_until}/{total_blocks} transformer blocks")
            
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        log.info(f"Model parameters: {trainable/1e6:.1f}M/{total/1e6:.1f}M trainable ({100*trainable/total:.1f}%)")
        
    def _setup_checkpointing(self):
        """Setup gradient checkpointing for memory efficiency"""
        
        if hasattr(self.base_model, 'joint_blocks'):
            for i, block in enumerate(self.base_model.joint_blocks):
                # Only checkpoint unfrozen blocks
                if any(p.requires_grad for p in block.parameters()):
                    # Wrap the forward method with checkpointing
                    original_forward = block.forward
                    
                    def make_checkpointed_forward(orig_forward):
                        def checkpointed_forward(*args, **kwargs):
                            if self.training:
                                return checkpoint(orig_forward, *args, **kwargs)
                            else:
                                return orig_forward(*args, **kwargs)
                        return checkpointed_forward
                    
                    block.forward = make_checkpointed_forward(original_forward)
                    
        log.info("Gradient checkpointing enabled for unfrozen blocks")
        
    def _setup_augmentation(self):
        """Setup data augmentation modules"""
        
        self.time_masking = TimeMasking(max_mask_ratio=0.1)
        self.frequency_masking = FrequencyMasking(max_mask_ratio=0.1)
        self.noise_injection = NoiseInjection(noise_level=0.01)
        self.mixup = MixUpAugmentation(alpha=0.2)
        
    def _apply_augmentation(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply data augmentation during training"""
        
        if not (self.training and self.enable_augmentation):
            return batch
            
        # Apply augmentations with some probability
        if torch.rand(1) < 0.5:
            batch = self.time_masking(batch)
            
        if torch.rand(1) < 0.3:
            batch = self.frequency_masking(batch)
            
        if torch.rand(1) < 0.3:
            batch = self.noise_injection(batch)
            
        if torch.rand(1) < 0.4:
            batch = self.mixup(batch)
            
        return batch
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with augmentation and mixed precision"""
        
        # Apply augmentation
        batch = self._apply_augmentation(batch)
        
        # Use autocast for mixed precision
        with autocast(enabled=True, dtype=torch.float16):
            # Extract inputs
            xt = batch.get('xt')
            if xt is None:
                # Construct xt from mean/std
                mean = batch['mean']
                std = batch['std']
                noise = torch.randn_like(mean)
                xt = mean + std * noise
                
            # Get conditioning
            clip_f = batch.get('clip_features')
            sync_f = batch.get('sync_features')
            text_f = batch.get('text_features')
            
            # Timestep (for flow matching)
            t = torch.rand(xt.shape[0], device=xt.device)
            
            # Forward through base model
            output = self.base_model(
                xt=xt,
                t=t,
                clip_f=clip_f,
                sync_f=sync_f,
                text_f=text_f
            )
            
            # Apply dropout for regularization
            output = self.dropout(output)
            
        return output
        
    def get_trainable_parameters(self):
        """Get only the trainable parameters for optimizer"""
        return filter(lambda p: p.requires_grad, self.parameters())
        
    def save_checkpoint(self, path: str, optimizer=None, scheduler=None, **kwargs):
        """Save checkpoint with additional metadata"""
        
        checkpoint = {
            'model_state_dict': self.base_model.state_dict(),
            'freeze_ratio': self.freeze_ratio,
            'enable_checkpointing': self.enable_checkpointing,
            'enable_augmentation': self.enable_augmentation,
            **kwargs
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        torch.save(checkpoint, path)
        log.info(f"Checkpoint saved to {path}")
        
    @classmethod
    def load_checkpoint(cls, path: str, base_model: MMAudio, optimizer=None, scheduler=None):
        """Load checkpoint and restore training state"""
        
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create wrapper with saved configuration
        wrapper = cls(
            base_model=base_model,
            freeze_ratio=checkpoint.get('freeze_ratio', 0.5),
            enable_checkpointing=checkpoint.get('enable_checkpointing', True),
            enable_augmentation=checkpoint.get('enable_augmentation', True)
        )
        
        # Load model weights
        wrapper.base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer/scheduler if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        log.info(f"Checkpoint loaded from {path}")
        return wrapper, checkpoint


# Data Augmentation Classes
class TimeMasking(nn.Module):
    """Apply time masking to latent sequences"""
    
    def __init__(self, max_mask_ratio: float = 0.1):
        super().__init__()
        self.max_mask_ratio = max_mask_ratio
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.training:
            return batch
            
        mean = batch.get('mean')
        if mean is None:
            return batch
            
        B, T, D = mean.shape
        max_mask_len = int(T * self.max_mask_ratio)
        
        if max_mask_len > 0:
            mask_len = torch.randint(1, max_mask_len + 1, (B,))
            mask_start = torch.randint(0, T - mask_len.max() + 1, (B,))
            
            for b in range(B):
                start = mask_start[b]
                end = start + mask_len[b]
                batch['mean'][b, start:end] = 0
                if 'std' in batch:
                    batch['std'][b, start:end] = 1
                    
        return batch


class FrequencyMasking(nn.Module):
    """Apply frequency masking to latent sequences"""
    
    def __init__(self, max_mask_ratio: float = 0.1):
        super().__init__()
        self.max_mask_ratio = max_mask_ratio
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.training:
            return batch
            
        mean = batch.get('mean')
        if mean is None:
            return batch
            
        B, T, D = mean.shape
        max_mask_len = int(D * self.max_mask_ratio)
        
        if max_mask_len > 0:
            mask_len = torch.randint(1, max_mask_len + 1, (B,))
            mask_start = torch.randint(0, D - mask_len.max() + 1, (B,))
            
            for b in range(B):
                start = mask_start[b]
                end = start + mask_len[b]
                batch['mean'][b, :, start:end] = 0
                    
        return batch


class NoiseInjection(nn.Module):
    """Inject small amount of noise for regularization"""
    
    def __init__(self, noise_level: float = 0.01):
        super().__init__()
        self.noise_level = noise_level
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.training:
            return batch
            
        for key in ['mean', 'clip_features', 'sync_features']:
            if key in batch:
                noise = torch.randn_like(batch[key]) * self.noise_level
                batch[key] = batch[key] + noise
                
        return batch


class MixUpAugmentation(nn.Module):
    """MixUp augmentation for better generalization"""
    
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.training or batch['mean'].shape[0] < 2:
            return batch
            
        B = batch['mean'].shape[0]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)  # Ensure lam > 0.5 for stability
        
        # Random permutation
        index = torch.randperm(B)
        
        # Mix features
        for key in ['mean', 'std', 'clip_features', 'sync_features', 'text_features']:
            if key in batch:
                mixed = lam * batch[key] + (1 - lam) * batch[key][index]
                batch[key] = mixed
                
        return batch