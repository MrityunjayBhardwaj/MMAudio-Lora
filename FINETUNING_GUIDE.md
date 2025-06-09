# MMAudio Fine-tuning Guide

This guide walks you through fine-tuning MMAudio on your custom 20 5-minute videos using a 70GB A100 GPU and 100GB RAM.

## Quick Start

```bash
# 1. Place your 20 videos in a directory
mkdir ./my_videos
# Copy your 20 5-minute videos here

# 2. Run the complete pipeline
./run_finetuning_pipeline.sh ./my_videos

# 3. Wait 24-48 hours for completion
```

## Prerequisites

### Hardware Requirements
- **GPU**: 70GB A100 (or equivalent with 50GB+ VRAM)
- **RAM**: 100GB system memory
- **Storage**: 50GB+ free space for features and checkpoints
- **Time**: 24-48 hours total

### Software Requirements
- Python 3.9+
- PyTorch 2.5.1+ with CUDA
- All MMAudio dependencies (see main README)

### Data Requirements
- 20 videos, approximately 5 minutes each
- Videos should have clear audio tracks
- Supported formats: MP4, AVI, MOV, MKV
- Consistent quality recommended

## Pipeline Overview

The fine-tuning pipeline consists of 6 steps:

1. **Video Splitting** (2-3 hours): Split 5-minute videos into 8-second clips with overlap
2. **Feature Extraction** (8-12 hours): Extract CLIP, Synchformer, and VAE features
3. **Configuration** (5 minutes): Setup training configurations
4. **Training** (24-36 hours): Fine-tune the model with memory optimizations
5. **Evaluation** (1-2 hours): Test the model and compare with original
6. **Sample Generation** (30 minutes): Generate sample outputs

## Step-by-Step Instructions

### Step 1: Prepare Your Data

```bash
# Create a directory for your videos
mkdir ./custom_videos

# Copy your 20 5-minute videos
cp /path/to/your/videos/*.mp4 ./custom_videos/

# Verify videos
ls -la ./custom_videos/
```

### Step 2: Run Video Splitting

```bash
python training/split_videos_for_finetuning.py \
    --input_dir ./custom_videos \
    --output_tsv ./custom_data/clips.tsv \
    --segment_duration 8.0 \
    --overlap 0.5
```

Expected output: ~1,500 clips with 50% overlap

### Step 3: Extract Features

```bash
# Single GPU (recommended for A100)
python training/extract_features_finetuning.py \
    --input_tsv ./custom_data/clips.tsv \
    --video_root ./custom_videos \
    --output_dir ./custom_data/features \
    --mode 16k \
    --batch_size 32 \
    --num_workers 24

# Multi-GPU (if available)
torchrun --nproc_per_node=2 training/extract_features_finetuning.py \
    --input_tsv ./custom_data/clips.tsv \
    --video_root ./custom_videos \
    --output_dir ./custom_data/features \
    --mode 16k \
    --batch_size 16 \
    --num_workers 12
```

This step takes 8-12 hours and creates ~1GB of feature data.

### Step 4: Start Fine-tuning

```bash
python train_finetune.py \
    exp_id=my_custom_finetune \
    model=small_16k \
    custom_finetune=true
```

**Training Configuration:**
- Model: small_16k (fits in 70GB VRAM)
- Batch size: 4 per GPU with gradient accumulation
- Learning rate: 1e-5 (low for stability)
- Max steps: 5,000 (~7 epochs)
- Early stopping: 3 evaluations without improvement

**Memory Optimizations:**
- 50% of transformer layers frozen
- Gradient checkpointing enabled
- Mixed precision (FP16) training
- Data augmentation for regularization

### Step 5: Monitor Training

Training logs are saved to `./output/my_custom_finetune/`. Monitor progress:

```bash
# View training logs
tail -f ./output/my_custom_finetune/train.log

# Monitor with tensorboard
tensorboard --logdir ./output/my_custom_finetune/
```

**Key metrics to watch:**
- Training loss should decrease steadily
- Validation loss should plateau (not decrease much)
- Look for signs of overfitting after ~2,000 steps

### Step 6: Evaluate Results

```bash
python evaluate_finetuned.py \
    --checkpoint ./output/my_custom_finetune/checkpoints/best_model.pt \
    --base_model small_16k \
    --test_videos ./custom_videos \
    --output_dir ./evaluation_results \
    --compare_original
```

This generates:
- Audio samples for test videos
- Comparison with original model
- Quality metrics and success rates

## Configuration Options

### Model Variants

```bash
# 16kHz model (recommended for 70GB GPU)
model=small_16k    # ~2B parameters, 16kHz audio

# 44kHz model (requires more memory)
model=small_44k    # ~2B parameters, 44.1kHz audio
```

### Training Hyperparameters

Key settings in `config/finetune_config.yaml`:

```yaml
# Memory management
batch_size: 4                    # Reduce if OOM
gradient_accumulation_steps: 8   # Effective batch size = 32
gradient_checkpointing: true     # Trade compute for memory

# Learning rate
learning_rate: 1e-5              # Conservative for fine-tuning
warmup_steps: 100               # Gradual warmup
max_steps: 5000                 # ~7 epochs with 1500 clips

# Regularization
freeze_ratio: 0.5               # Freeze 50% of layers
dropout_rate: 0.2               # Prevent overfitting
weight_decay: 0.01              # L2 regularization

# Early stopping
early_stopping:
  patience: 3                   # Stop if no improvement
  min_delta: 0.001             # Minimum improvement threshold
```

### Data Augmentation

```yaml
augmentation:
  enabled: true
  time_masking: true            # Temporal dropout
  freq_masking: true            # Spectral dropout
  mixup_alpha: 0.2             # MixUp for generalization
  noise_injection: 0.01        # Small noise for robustness
```

## Troubleshooting

### Memory Issues

**GPU Out of Memory:**
```bash
# Reduce batch size
batch_size: 2
gradient_accumulation_steps: 16

# Increase gradient checkpointing
gradient_checkpointing: true

# Use smaller model
model: small_16k  # instead of medium/large
```

**System RAM Issues:**
```bash
# Reduce data loading workers
num_workers: 8  # instead of 24

# Disable memory pinning
pin_memory: false
```

### Training Issues

**Loss not decreasing:**
- Check if learning rate is too low (try 2e-5)
- Verify data loading is working correctly
- Check for gradient clipping issues

**Overfitting quickly:**
- Increase dropout rate to 0.3
- Reduce learning rate to 5e-6
- Add more aggressive augmentation

**Training unstable:**
- Disable model compilation: `compile: false`
- Use smaller batch size
- Check for NaN values in gradients

### Data Issues

**Feature extraction fails:**
```bash
# Check video formats are supported
ffprobe your_video.mp4

# Verify audio tracks exist
ffmpeg -i your_video.mp4 -vn -acodec copy audio_check.wav

# Reduce batch size if OOM during extraction
--batch_size 16
```

**Poor quality results:**
- Ensure videos have clear, prominent audio
- Check that audio matches visual content
- Verify video resolution and quality

## Expected Results

### Success Metrics

**Training Success:**
- Training loss decreases from ~2.0 to ~0.5
- Validation loss plateaus around 0.6-0.8
- No signs of severe overfitting
- Training completes without OOM errors

**Quality Improvements:**
- Generated audio matches video content better
- 15-20% improvement on domain-specific videos
- Maintains general capabilities on other content
- Audio is temporally coherent and realistic

### Realistic Expectations

**Best Case (60% probability):**
- Model adapts well to your video style
- Clear improvement on similar content
- Good temporal synchronization
- Reasonable generalization

**Likely Case (30% probability):**
- Modest improvement over base model
- Some overfitting but usable results
- Works well only on very similar videos
- Quality varies across different content

**Worst Case (10% probability):**
- Catastrophic forgetting of general skills
- Severe overfitting to training data
- Poor audio quality overall
- Need to restart with different hyperparameters

## Performance Optimization

### For Faster Training

```bash
# Enable compilation (if stable)
compile: true

# Use larger batch sizes if memory allows
batch_size: 8
gradient_accumulation_steps: 4

# Reduce validation frequency
eval_every: 1000

# Use fewer workers for data loading
num_workers: 16
```

### For Better Quality

```bash
# More training steps
max_steps: 10000

# Lower learning rate
learning_rate: 5e-6

# More aggressive regularization
dropout_rate: 0.3
weight_decay: 0.02

# Enable all augmentations
augmentation:
  enabled: true
  time_masking: true
  freq_masking: true
  mixup_alpha: 0.3
  noise_injection: 0.02
```

## Advanced Usage

### Custom Prompts

Create a prompts file for better control:

```csv
video_name,prompt
video1,sound of ocean waves and seagulls
video2,urban traffic and city ambience
video3,natural forest sounds with birds
```

Use with evaluation:
```bash
python evaluate_finetuned.py \
    --prompts_file prompts.csv \
    --checkpoint best_model.pt \
    --test_videos ./test_videos
```

### Adapter-based Fine-tuning

For even more memory efficiency, consider adapter-based training:

```python
# In finetune_config.yaml
use_adapters: true
adapter_dim: 64
freeze_base_model: true
```

### Progressive Training

Start with shorter sequences and gradually increase:

```bash
# Phase 1: 4-second clips
python train_finetune.py segment_duration=4.0 max_steps=2000

# Phase 2: 8-second clips  
python train_finetune.py segment_duration=8.0 max_steps=5000 \
    checkpoint=./output/phase1/best_model.pt
```

## File Structure

After completion, your directory will look like:

```
MMAudio/
├── custom_videos/           # Your input videos
├── custom_data/
│   ├── clips.tsv           # Video clip metadata
│   ├── features/           # Extracted features (1GB)
│   └── best_model_path.txt # Path to trained model
├── output/
│   └── my_custom_finetune/
│       ├── checkpoints/    # Model checkpoints
│       ├── logs/          # Training logs
│       └── config.yaml    # Used configuration
└── evaluation_results/
    ├── generated_audio/   # Generated samples
    ├── comparison/       # Original vs fine-tuned
    └── results.csv       # Evaluation metrics
```

## Next Steps

After successful fine-tuning:

1. **Test on new videos** similar to your training data
2. **Compare with original model** on various content types
3. **Adjust hyperparameters** if results aren't satisfactory
4. **Consider ensemble methods** combining original and fine-tuned models
5. **Collect more data** if quality improvements are insufficient

## Support

For issues:
1. Check the troubleshooting section above
2. Verify hardware requirements are met
3. Review training logs for error messages
4. Test with smaller datasets first
5. Open issues on the repository for bugs

Remember: Fine-tuning on 20 videos is challenging and results may vary. The small dataset size relative to model capacity makes overfitting likely, but the memory optimizations and regularization techniques should help achieve reasonable results.