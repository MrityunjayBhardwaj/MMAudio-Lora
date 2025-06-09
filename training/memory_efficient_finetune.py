import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import logging

log = logging.getLogger(__name__)

class MemoryEfficientMMAudio(nn.Module):
    """Wrapper for MMAudio with memory optimizations"""
    
    def __init__(self, base_model, enable_checkpointing=True):
        super().__init__()
        self.base_model = base_model
        self.enable_checkpointing = enable_checkpointing
        
        # Freeze early layers to save memory
        self._freeze_early_layers()
        
    def _freeze_early_layers(self):
        """Freeze feature extractors and early transformer blocks"""
        # Freeze input projections (they're already well-trained)
        for param in self.base_model.audio_input_proj.parameters():
            param.requires_grad = False
        for param in self.base_model.clip_input_proj.parameters():
            param.requires_grad = False
        for param in self.base_model.sync_input_proj.parameters():
            param.requires_grad = False
            
        # Freeze first 50% of transformer blocks
        total_blocks = len(self.base_model.joint_blocks)
        freeze_until = total_blocks // 2
        
        for i in range(freeze_until):
            for param in self.base_model.joint_blocks[i].parameters():
                param.requires_grad = False
                
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        log.info(f"Froze early layers: {trainable/1e6:.1f}M/{total/1e6:.1f}M parameters trainable")
        
    def forward(self, *args, **kwargs):
        if self.enable_checkpointing and self.training:
            # Use gradient checkpointing for memory efficiency
            return self._checkpointed_forward(*args, **kwargs)
        else:
            return self.base_model(*args, **kwargs)
            
    def _checkpointed_forward(self, *args, **kwargs):
        # Implement gradient checkpointing for joint blocks
        # This trades compute for memory
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
            
        # Only checkpoint the blocks that are being trained
        for block in self.base_model.joint_blocks:
            if any(p.requires_grad for p in block.parameters()):
                # Checkpoint this block
                block.forward = checkpoint(create_custom_forward(block), block.forward)
                
        return self.base_model(*args, **kwargs)


class FineTuningOptimizer:
    """Optimized training loop for fine-tuning"""
    
    def __init__(self, model, lr=1e-5, warmup_steps=100):
        self.model = model
        self.scaler = GradScaler()
        
        # Use AdamW with lower memory footprint
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            betas=(0.9, 0.999),  # Standard betas
            eps=1e-8,
            weight_decay=0.01,
            fused=True  # A100 supports fused AdamW
        )
        
        # Cosine annealing with warmup
        self.warmup_steps = warmup_steps
        self.scheduler = self._create_scheduler()
        
    def _create_scheduler(self):
        # Implement warmup + cosine annealing
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                # Cosine annealing after warmup
                progress = (step - self.warmup_steps) / (total_steps - self.warmup_steps)
                return 0.5 * (1 + torch.cos(torch.pi * progress))
                
        return LambdaLR(self.optimizer, lr_lambda)
        
    def training_step(self, batch, accumulation_steps=4):
        """Single training step with gradient accumulation"""
        
        with autocast(dtype=torch.float16):
            loss = self.model(batch) / accumulation_steps
            
        self.scaler.scale(loss).backward()
        
        if (self.step + 1) % accumulation_steps == 0:
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                max_norm=1.0
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
        return loss.item() * accumulation_steps