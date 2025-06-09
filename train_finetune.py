#!/usr/bin/env python3
"""
Fine-tuning script for MMAudio on custom video datasets.
Optimized for A100 GPU with memory-efficient training.
"""

import logging
import math
import random
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as distributed
import numpy as np
from torch.cuda.amp import GradScaler
from torch.utils.data import random_split
from omegaconf import DictConfig, open_dict
import hydra
from hydra.core.hydra_config import HydraConfig

from mmaudio.data.data_setup import setup_training_datasets
from mmaudio.model.networks import get_my_mmaudio
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.finetune_wrapper import MemoryEfficientMMAudio
from mmaudio.model.sequence_config import CONFIG_16K, CONFIG_44K
from mmaudio.utils.dist_utils import info_if_rank_zero, local_rank, world_size
from mmaudio.utils.logger import TensorboardLogger
from mmaudio.eval_utils import all_model_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


class FineTuningTrainer:
    """Fine-tuning trainer with memory optimizations and early stopping"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(f'cuda:{local_rank}')
        self.run_dir = HydraConfig.get().run.dir
        
        # Initialize components
        self.setup_model()
        self.setup_optimizer()
        self.setup_data()
        self.setup_training_state()
        
    def setup_model(self):
        """Initialize model with pretrained weights and memory optimizations"""
        
        # Get model configuration
        model_name = self.cfg.model
        if model_name not in all_model_cfg:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_config = all_model_cfg[model_name]
        model_config.download_if_needed()
        
        # Load base model
        base_model = get_my_mmaudio(model_name).to(self.device)
        
        # Load pretrained weights
        checkpoint = torch.load(model_config.model_path, map_location=self.device)
        base_model.load_state_dict(checkpoint, strict=True)
        log.info(f"Loaded pretrained weights from {model_config.model_path}")
        
        # Wrap with memory-efficient wrapper
        self.model = MemoryEfficientMMAudio(
            base_model=base_model,
            freeze_ratio=self.cfg.get('freeze_ratio', 0.5),
            enable_checkpointing=self.cfg.get('gradient_checkpointing', True),
            dropout_rate=self.cfg.get('dropout_rate', 0.1),
            enable_augmentation=self.cfg.augmentation.get('enabled', True)
        ).to(self.device)
        
        # Initialize flow matching
        self.flow_matching = FlowMatching(
            min_sigma=0.0,
            inference_mode='euler',
            num_steps=25
        )
        
        log.info("Model initialized with memory optimizations")
        
    def setup_optimizer(self):
        """Setup optimizer with gradient scaling and scheduling"""
        
        # Get only trainable parameters
        trainable_params = list(self.model.get_trainable_parameters())
        
        # AdamW optimizer with fused implementation for A100
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.cfg.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.cfg.weight_decay,
            fused=True  # A100 supports fused AdamW
        )
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler(enabled=self.cfg.amp)
        
        # Learning rate scheduler
        self.scheduler = self.create_scheduler()
        
        log.info(f"Optimizer configured for {len(trainable_params)} parameters")
        
    def create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        
        from torch.optim.lr_scheduler import LambdaLR
        
        warmup_steps = self.cfg.warmup_steps
        max_steps = self.cfg.max_steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
                
        return LambdaLR(self.optimizer, lr_lambda)
        
    def setup_data(self):
        """Setup training and validation datasets"""
        
        # Patch data dimensions based on model
        if self.cfg.model.endswith('16k'):
            seq_cfg = CONFIG_16K
        elif self.cfg.model.endswith('44k'):
            seq_cfg = CONFIG_44K
        else:
            raise ValueError(f'Unknown model: {self.cfg.model}')
            
        with open_dict(self.cfg):
            self.cfg.data_dim.latent_seq_len = seq_cfg.latent_seq_len
            self.cfg.data_dim.clip_seq_len = seq_cfg.clip_seq_len
            self.cfg.data_dim.sync_seq_len = seq_cfg.sync_seq_len
        
        # Load full dataset
        full_dataset, _, _ = setup_training_datasets(self.cfg)
        
        # Split into train/validation
        if self.cfg.validation.enabled:
            val_size = int(len(full_dataset) * self.cfg.validation.split_ratio)
            train_size = len(full_dataset) - val_size
            
            generator = torch.Generator().manual_seed(self.cfg.validation.random_seed)
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size], generator=generator
            )
        else:
            self.train_dataset = full_dataset
            self.val_dataset = None
            
        # Create data loaders
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        
        train_sampler = DistributedSampler(self.train_dataset, rank=local_rank, shuffle=True)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=train_sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.hardware.get('prefetch_factor', 2),
            persistent_workers=self.cfg.hardware.get('persistent_workers', True)
        )
        
        if self.val_dataset is not None:
            val_sampler = DistributedSampler(self.val_dataset, rank=local_rank, shuffle=False)
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.batch_size * 2,  # Larger batch for validation
                sampler=val_sampler,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory
            )
        else:
            self.val_loader = None
            
        log.info(f"Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            log.info(f"Validation samples: {len(self.val_dataset)}")
            
    def setup_training_state(self):
        """Initialize training state and logging"""
        
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Setup logger
        self.logger = TensorboardLogger(
            self.cfg.exp_id,
            self.run_dir,
            logging.getLogger(),
            is_rank0=(local_rank == 0),
            enable_email=False
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.run_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def training_step(self, batch):
        """Single training step with gradient accumulation"""
        
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Prepare flow matching inputs
        mean = batch['mean']
        std = batch['std']
        
        # Sample noise and timestep
        x0 = torch.randn_like(mean)
        x1 = mean + std * torch.randn_like(std)
        t = torch.rand(mean.shape[0], device=self.device)
        
        # Get conditional flow
        xt = self.flow_matching.get_conditional_flow(x0, x1, t.unsqueeze(-1).unsqueeze(-1))
        
        # Add to batch
        batch['xt'] = xt
        batch['t'] = t
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast(enabled=self.cfg.amp):
            # Model prediction
            predicted_v = self.model(batch)
            
            # Flow matching loss
            loss = self.flow_matching.loss(predicted_v, x0, x1)
            loss = loss.mean()  # Average over batch
            
            # Scale loss for gradient accumulation
            loss = loss / self.cfg.gradient_accumulation_steps
            
        # Backward pass
        self.scaler.scale(loss).backward()
        
        return loss.item() * self.cfg.gradient_accumulation_steps
        
    def validation_step(self):
        """Validation loop"""
        
        if self.val_loader is None:
            return None
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Prepare inputs
                mean = batch['mean']
                std = batch['std']
                x0 = torch.randn_like(mean)
                x1 = mean + std * torch.randn_like(std)
                t = torch.rand(mean.shape[0], device=self.device)
                xt = self.flow_matching.get_conditional_flow(x0, x1, t.unsqueeze(-1).unsqueeze(-1))
                
                batch['xt'] = xt
                batch['t'] = t
                
                with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                    predicted_v = self.model(batch)
                    loss = self.flow_matching.loss(predicted_v, x0, x1).mean()
                    
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
        
    def check_early_stopping(self, val_loss):
        """Check early stopping condition"""
        
        if not self.cfg.early_stopping.enabled:
            return False
            
        min_delta = self.cfg.early_stopping.min_delta
        patience = self.cfg.early_stopping.patience
        
        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            
            # Save best model
            self.save_checkpoint('best_model.pt', is_best=True)
            return False
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= patience:
                log.info(f"Early stopping triggered after {patience} evaluations without improvement")
                return True
                
        return False
        
    def save_checkpoint(self, filename, is_best=False):
        """Save training checkpoint"""
        
        checkpoint_path = self.checkpoint_dir / filename
        
        self.model.save_checkpoint(
            path=str(checkpoint_path),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            epoch=self.epoch,
            best_val_loss=self.best_val_loss,
            config=self.cfg,
            is_best=is_best
        )
        
        info_if_rank_zero(self.logger, f"Checkpoint saved: {filename}")
        
    def train(self):
        """Main training loop"""
        
        info_if_rank_zero(self.logger, "Starting fine-tuning...")
        
        while self.step < self.cfg.max_steps:
            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.epoch)
                
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                loss = self.training_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # Gradient accumulation
                if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
                    # Unscale gradients for clipping
                    self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_trainable_parameters(),
                        max_norm=1.0
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.step += 1
                    
                    # Logging
                    if self.step % self.cfg.logging.log_every == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        info_if_rank_zero(
                            self.logger,
                            f"Step {self.step}/{self.cfg.max_steps}, "
                            f"Loss: {loss:.4f}, LR: {lr:.2e}"
                        )
                        
                        if hasattr(self.logger, 'log_scalar'):
                            self.logger.log_scalar('train/loss', loss, self.step)
                            self.logger.log_scalar('train/lr', lr, self.step)
                    
                    # Validation
                    if self.step % self.cfg.eval_every == 0:
                        val_loss = self.validation_step()
                        
                        if val_loss is not None:
                            info_if_rank_zero(
                                self.logger,
                                f"Validation loss: {val_loss:.4f}"
                            )
                            
                            if hasattr(self.logger, 'log_scalar'):
                                self.logger.log_scalar('val/loss', val_loss, self.step)
                            
                            # Early stopping check
                            if self.check_early_stopping(val_loss):
                                return
                                
                    # Checkpointing
                    if self.step % self.cfg.save_every == 0:
                        self.save_checkpoint(f'checkpoint_step_{self.step}.pt')
                        
                    # Check if training is complete
                    if self.step >= self.cfg.max_steps:
                        break
                        
                # Memory cleanup
                if self.step % 100 == 0:
                    torch.cuda.empty_cache()
                    
            self.epoch += 1
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            info_if_rank_zero(
                self.logger,
                f"Epoch {self.epoch} completed. Average loss: {avg_epoch_loss:.4f}"
            )
            
        # Final checkpoint
        self.save_checkpoint('final_model.pt')
        info_if_rank_zero(self.logger, "Fine-tuning completed!")


def distributed_setup():
    """Initialize distributed training"""
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')


@hydra.main(version_base='1.3.2', config_path='config', config_name='finetune_config.yaml')
def main(cfg: DictConfig):
    """Main entry point"""
    
    # Setup
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    
    if world_size > 1:
        distributed_setup()
        
    # Set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    # Initialize trainer
    trainer = FineTuningTrainer(cfg)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()