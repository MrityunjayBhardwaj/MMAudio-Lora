#!/usr/bin/env python3
"""
Optimized feature extraction for fine-tuning with memory efficiency.
Designed for A100 GPU with mixed precision and larger batch sizes.
"""

import logging
import os
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import pandas as pd
import tensordict as td
import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.cuda.amp import autocast

from mmaudio.data.extraction.vgg_sound import VGGSound
from mmaudio.model.utils.features_utils import FeaturesUtils
from mmaudio.utils.dist_utils import local_rank, world_size

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()
log.setLevel(logging.INFO)


class OptimizedFeatureExtractor:
    """Memory-efficient feature extractor for fine-tuning"""
    
    def __init__(self, mode='16k', batch_size=32, num_workers=24):
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Configure based on mode
        if mode == '16k':
            self.sampling_rate = 16000
            self.num_samples = 128000
            self.vae_path = './ext_weights/v1-16.pth'
            self.bigvgan_path = './ext_weights/best_netG.pt'
        else:  # 44k
            self.sampling_rate = 44100
            self.num_samples = 353280
            self.vae_path = './ext_weights/v1-44.pth'
            self.bigvgan_path = None
            
        self.synchformer_ckpt = './ext_weights/synchformer_state_dict.pth'
        self.duration_sec = 8.0
        
        # Initialize device
        self.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device)
        
    def setup_models(self):
        """Initialize feature extraction models with optimizations"""
        
        log.info("Loading feature extraction models...")
        
        # Initialize with half precision for memory efficiency
        self.feature_utils = FeaturesUtils(
            tod_vae_ckpt=self.vae_path,
            synchformer_ckpt=self.synchformer_ckpt,
            enable_conditions=True,
            mode=self.mode,
            bigvgan_vocoder_ckpt=self.bigvgan_path,
            need_vae_encoder=True
        ).to(self.device, torch.float16).eval()
        
        # Compile for faster inference on A100
        if hasattr(torch, 'compile'):
            log.info("Compiling models for faster inference...")
            self.feature_utils.compile()
            
        log.info("Models loaded and optimized")
        
    def extract_batch(self, batch):
        """Extract features from a batch with mixed precision"""
        
        with autocast(dtype=torch.float16):
            # Move batch to device
            waveform = batch['waveform'].to(self.device, non_blocking=True)
            clip_video = batch['clip_video'].to(self.device, non_blocking=True)
            sync_video = batch['sync_video'].to(self.device, non_blocking=True)
            caption = batch['caption']
            
            # Extract audio features
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)
                
            mean, std = self.feature_utils.encode(waveform)
            
            # Extract visual features with larger batch size multiplier
            clip_features = self.feature_utils.encode_video_with_clip(
                clip_video, batch_size_multiplier=60  # Increased for A100
            )
            sync_features = self.feature_utils.encode_video_with_sync(
                sync_video, batch_size_multiplier=60
            )
            
            # Extract text features
            text_features = self.feature_utils.encode_text(caption)
            
        # Convert back to float32 for storage
        return {
            'mean': mean.float().cpu(),
            'std': std.float().cpu(),
            'clip_features': clip_features.float().cpu(),
            'sync_features': sync_features.float().cpu(),
            'text_features': text_features.float().cpu()
        }
        
    def process_dataset(self, dataset_config, output_dir):
        """Process entire dataset with progress tracking"""
        
        # Load dataset
        dataset = VGGSound(
            dataset_config['root'],
            tsv_path=dataset_config['tsv_path'],
            sample_rate=self.sampling_rate,
            duration_sec=self.duration_sec,
            audio_samples=self.num_samples,
            normalize_audio=dataset_config.get('normalize_audio', True)
        )
        
        # Create distributed sampler
        sampler = DistributedSampler(dataset, rank=local_rank, shuffle=False)
        
        # Create data loader with optimized settings
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=False,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        # Prepare output storage
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get dataset info
        df = pd.read_csv(dataset_config['tsv_path'], sep='\t')
        total_samples = len(df)
        
        # Pre-allocate tensors for efficiency
        log.info(f"Pre-allocating tensors for {total_samples} samples...")
        
        # Determine dimensions
        with torch.no_grad():
            sample_batch = next(iter(loader))
            sample_features = self.extract_batch(sample_batch)
            
        # Create memory-mapped tensors
        mmap_dict = {
            'mean': torch.zeros(
                (total_samples, *sample_features['mean'].shape[1:]),
                dtype=torch.float32
            ),
            'std': torch.zeros(
                (total_samples, *sample_features['std'].shape[1:]),
                dtype=torch.float32
            ),
            'clip_features': torch.zeros(
                (total_samples, *sample_features['clip_features'].shape[1:]),
                dtype=torch.float32
            ),
            'sync_features': torch.zeros(
                (total_samples, *sample_features['sync_features'].shape[1:]),
                dtype=torch.float32
            ),
            'text_features': torch.zeros(
                (total_samples, *sample_features['text_features'].shape[1:]),
                dtype=torch.float32
            )
        }
        
        # Process batches
        log.info("Starting feature extraction...")
        
        idx = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting features"):
                batch_size = batch['waveform'].shape[0]
                
                # Extract features
                features = self.extract_batch(batch)
                
                # Store in memory-mapped tensors
                for key, value in features.items():
                    mmap_dict[key][idx:idx + batch_size] = value
                    
                idx += batch_size
                
                # Clear cache periodically to prevent memory buildup
                if idx % 1000 == 0:
                    torch.cuda.empty_cache()
                    
        # Save as TensorDict
        log.info(f"Saving features to {output_path}...")
        
        td_save = td.TensorDict(mmap_dict, batch_size=[total_samples])
        td_save.memmap_(output_path)
        
        log.info(f"Features saved successfully to {output_path}")
        
        # Save metadata TSV
        metadata_path = output_path.parent / f"{output_path.name}.tsv"
        df.to_csv(metadata_path, sep='\t', index=False)
        
        return output_path


def distributed_setup():
    """Initialize distributed training"""
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
    return local_rank, world_size


def main():
    parser = ArgumentParser(description='Extract features for fine-tuning')
    
    parser.add_argument(
        '--input_tsv',
        type=str,
        required=True,
        help='TSV file with clip metadata'
    )
    parser.add_argument(
        '--video_root',
        type=str,
        required=True,
        help='Root directory containing videos'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./custom_data/features',
        help='Output directory for features'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['16k', '44k'],
        default='16k',
        help='Audio mode (16k or 44k)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=24,
        help='Number of data loading workers'
    )
    
    args = parser.parse_args()
    
    # Setup distributed if available
    if 'WORLD_SIZE' in os.environ:
        distributed_setup()
        
    # Initialize extractor
    extractor = OptimizedFeatureExtractor(
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Setup models
    extractor.setup_models()
    
    # Process dataset
    dataset_config = {
        'root': args.video_root,
        'tsv_path': args.input_tsv,
        'normalize_audio': True
    }
    
    output_path = extractor.process_dataset(dataset_config, args.output_dir)
    
    log.info(f"Feature extraction complete! Output saved to {output_path}")
    
    # Print summary statistics
    if local_rank == 0:
        import tensordict
        td_loaded = tensordict.TensorDict.load_memmap(output_path)
        
        print("\n=== Feature Summary ===")
        for key, tensor in td_loaded.items():
            print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
            

if __name__ == '__main__':
    main()