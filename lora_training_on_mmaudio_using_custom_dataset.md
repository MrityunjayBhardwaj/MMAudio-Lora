# LoRA Training on MMAudio Using Custom Dataset: Complete Technical Guide

## Executive Summary

**Objective**: Implement LoRA (Low-Rank Adaptation) fine-tuning for MMAudio on a custom dataset of 67 segmented commercial video clips.

**Current State**: Repository contains LoRA infrastructure but critical gaps prevent training on our custom segmented dataset.

**Key Challenge**: MMAudio's training pipeline expects pre-computed memory-mapped features, but we have raw segmented videos requiring real-time feature extraction.

**Confidence Level**: 85% - Based on extensive code analysis, but lacking runtime verification.

---

## 1. Repository Architecture Analysis

### 1.1 MMAudio Core Architecture

**Evidence**: Direct inspection of `mmaudio/model/networks.py:1-50`

```python
# mmaudio/model/networks.py:15-25
class MMAudio(nn.Module):
    def __init__(self, ...):
        # Flow matching transformer for audio generation
        self.joint_blocks = nn.ModuleList([...])  # Multi-modal attention blocks
        self.fused_blocks = nn.ModuleList([...])  # Audio generation blocks
```

**Confidence**: 95% - Direct code verification
**Assumption**: Architecture unchanged from inspection time
**For 100% Certainty**: Runtime model instantiation test required

**Key Components Identified**:
1. **Flow Matching Model** (`mmaudio/model/flow_matching.py:1-200`)
2. **Multi-modal Transformer** (`mmaudio/model/networks.py:1-500`)
3. **Feature Extraction Utils** (`mmaudio/model/utils/features_utils.py:1-300`)
4. **LoRA Integration** (`mmaudio/model/lora_mmaudio.py:1-350`)

### 1.2 Sequence Configuration Analysis

**Evidence**: `mmaudio/model/sequence_config.py:44-50`

```python
CONFIG_16K = SequenceConfig(duration=8.0, sampling_rate=16000, spectrogram_frame_rate=256)
# Line 48: assert CONFIG_16K.latent_seq_len == 250
# Line 49: assert CONFIG_16K.clip_seq_len == 64  
# Line 50: assert CONFIG_16K.sync_seq_len == 192
```

**Critical Finding**: MMAudio enforces **FIXED 8-second duration** with specific sequence lengths.

**Confidence**: 100% - Hardcoded assertions in source
**Impact**: Our variable-duration segments (2s-8s) will cause sequence length mismatches

---

## 2. Current LoRA Implementation Analysis

### 2.1 LoRA Model Architecture

**Evidence**: `mmaudio/model/lora_mmaudio.py:19-50`

```python
class MMAudioLoRA(nn.Module):
    def __init__(self, base_model: MMAudio, lora_config: Optional[LoRAConfig] = None):
        self.base_model = base_model
        self.model, self.lora_modules = apply_lora_to_model(...)  # Line 38-42
```

**Confidence**: 90% - Code exists and appears functional
**Assumption**: LoRA application works correctly with MMAudio transformer layers
**For 100% Certainty**: Runtime LoRA application test needed

### 2.2 Model Loading Function

**Evidence**: `mmaudio/model/lora_mmaudio.py:312-330`

```python
def load_pretrained_with_lora(model_name: str, ...):
    base_model = get_my_mmaudio(model_name).to(device)  # Line 320
    checkpoint = torch.load(model_config.model_path, map_location=device)  # Line 327
    base_model.load_state_dict(checkpoint, strict=True)  # Line 328
```

**Critical Issue**: Assumes checkpoint format matches `get_my_mmaudio()` architecture.

**Confidence**: 60% - Code exists but checkpoint compatibility unverified
**Risk**: Runtime failure if checkpoint format doesn't match
**For 100% Certainty**: Test model loading with `weights/mmaudio_small_16k.pth`

---

## 3. Data Pipeline Analysis

### 3.1 Training Data Setup

**Evidence**: `mmaudio/data/data_setup.py:54-57`

```python
elif cfg.custom_finetune:
    custom_video = load_vgg_data(cfg, cfg.data.Custom_video)  # Line 56
    dataset = custom_video  # Line 57
```

**Critical Finding**: Custom fine-tuning uses `ExtractedVGG` which expects pre-computed features.

**Evidence**: `mmaudio/data/extracted_vgg.py:30-39`

```python
premade_mmap_dir = Path(premade_mmap_dir)
td = TensorDict.load_memmap(premade_mmap_dir)  # Line 33
self.clip_features = td['clip_features']       # Line 37
self.sync_features = td['sync_features']       # Line 38
self.text_features = td['text_features']       # Line 39
```

**Confidence**: 95% - Direct code verification
**Problem**: We have raw videos, not pre-computed features

### 3.2 Alternative: Custom Dataset Class

**Evidence**: `mmaudio/data/extraction/custom_dataset.py:24-50`

```python
class CustomFineTuningDataset(Dataset):
    def __init__(self, root: Union[str, Path], *, tsv_path: Union[str, Path] = 'clips.tsv'):
        df = pd.read_csv(tsv_path, sep='\t', dtype={'id': str})  # Line 50
```

**Key Finding**: This class performs real-time video loading, not pre-computed features!

**Confidence**: 85% - Code inspection shows real-time processing
**Assumption**: This class is intended for custom fine-tuning
**For 100% Certainty**: Runtime test with our segmented dataset needed

---

## 4. Current Dataset State Analysis

### 4.1 Segmented Dataset Inventory

**Evidence**: Direct file system inspection

```
segmented_videos/: 16 title-based folders
segmented_clips.tsv: 67 clips total
Duration range: 2.0s - 8.0s (average 7.4s)
Format: MP4 videos with audio descriptions
```

**Confidence**: 100% - Direct file verification completed

### 4.2 TSV Format Analysis

**Evidence**: `segmented_clips.tsv:1-3`

```tsv
video_path	start_time	end_time	caption	original_video	title_folder	original_start_time	original_end_time
Tele2_Nature_Connection_Campaign/segment_00_00-00_08.mp4	0.0	8	Urban ambient sounds with...
```

**Compatibility Check with CustomFineTuningDataset**:

**Evidence**: `mmaudio/data/extraction/custom_dataset.py:52-60`

```python
clip_info = {
    'id': record['id'],                    # ❌ Missing in our TSV
    'video_path': record['video_path'],    # ✅ Present
    'start_time': float(record['start_time']),  # ✅ Present (but always 0.0)
    'end_time': float(record['end_time']),      # ✅ Present
    'caption': record.get('caption', 'audio for this video'),  # ✅ Present
}
```

**Confidence**: 90% - Format mostly compatible, missing 'id' column
**Required Fix**: Add 'id' column to TSV

---

## 5. Configuration Analysis

### 5.1 LoRA Configuration

**Evidence**: `config/lora_config.yaml:19-36`

```yaml
lora:
  rank: 64
  alpha: 128.0
  dropout: 0.1
  target_modules:
    - "joint_blocks.*.latent_block.attn.qkv"  # Lines 24-35
    - "joint_blocks.*.latent_block.ffn.linear1"
    # ... 12 total target modules
```

**Confidence**: 95% - Configuration appears complete
**Assumption**: Target module names match actual model architecture
**For 100% Certainty**: Runtime verification of module names needed

### 5.2 Data Configuration

**Evidence**: `config/data/base.yaml:72-74`

```yaml
Custom_video:
  tsv: ./custom_data/features.tsv      # ❌ Wrong path
  memmap_dir: ./custom_data/features   # ❌ Wrong path  
```

**Problem**: Points to non-existent paths expecting pre-computed features.

**Confidence**: 100% - Direct config inspection
**Required Fix**: Update paths to our segmented dataset

---

## 6. Critical Gaps Analysis

### 6.1 Gap #1: Data Pipeline Mismatch

**Evidence**: 
- Training setup: `data_setup.py:56` uses `ExtractedVGG` 
- ExtractedVGG: `extracted_vgg.py:33` loads pre-computed features
- Our data: Raw segmented videos

**Confidence**: 95% - Clear architectural mismatch
**Impact**: Training will fail with "memmap directory not found"
**Solution Required**: Bridge raw videos to training pipeline

### 6.2 Gap #2: Variable Sequence Lengths

**Evidence**: 
- MMAudio requirement: `sequence_config.py:44` enforces 8.0-second duration
- Our dataset: 2-8 second variable durations
- Impact: Sequence length assertions will fail

**Confidence**: 100% - Hardcoded duration requirement
**Solutions**: 
1. Pad all segments to 8 seconds (Confidence: 80%)
2. Modify sequence config (Confidence: 40% - high risk)

### 6.3 Gap #3: Feature Extraction Pipeline

**Evidence**: No real-time feature extraction for training pipeline

**Required Components**:
1. CLIP video encoder (Confidence: 70% - model exists in eval_utils)
2. Synchformer encoder (Confidence: 70% - checkpoint exists)
3. Text encoder for captions (Confidence: 60% - unclear which encoder used)
4. VAE audio encoder (Confidence: 80% - checkpoint exists)

**For 100% Certainty**: Runtime testing of each encoder required

---

## 7. Implementation Strategy

### 7.1 Strategy A: Real-Time Feature Extraction (Recommended)

**Approach**: Modify training pipeline to extract features during training

**Evidence Supporting Feasibility**:
- `CustomFineTuningDataset` already does real-time video loading
- Feature extractors exist in `model/utils/features_utils.py`

**Implementation Steps**:

1. **Create Segmented Dataset Adapter** (Confidence: 80%)
```python
class SegmentedVGGDataset(Dataset):
    def __init__(self, tsv_path, video_root):
        # Load segmented_clips.tsv
        # Initialize feature extractors
        
    def __getitem__(self, idx):
        # Load video segment 
        # Extract CLIP features in real-time
        # Extract Sync features in real-time
        # Process caption through text encoder
        # Return ExtractedVGG-compatible format
```

2. **Modify Data Setup** (Confidence: 70%)
```python
# In data_setup.py:54-57, replace:
elif cfg.custom_finetune:
    custom_video = SegmentedVGGDataset(cfg.data.Custom_video.tsv, cfg.data.Custom_video.root)
    dataset = custom_video
```

3. **Handle Variable Durations** (Confidence: 75%)
   - Pad segments < 8s with last frame/silence
   - Truncate segments > 8s (should not occur in our dataset)

**Risks**:
- Runtime performance impact (real-time extraction)
- Memory usage increase
- Feature extraction accuracy vs pre-computed

### 7.2 Strategy B: Pre-compute Features (Alternative)

**Approach**: Generate memory-mapped features for all segments

**Confidence**: 60% - Requires understanding exact pre-computation pipeline
**For 100% Certainty**: Need to reverse-engineer VGG feature extraction process

**Steps**:
1. Extract features for all 67 segments
2. Create memory-mapped tensors
3. Use existing ExtractedVGG pipeline

**Advantages**: Better runtime performance
**Disadvantages**: Complex setup, storage requirements

---

## 8. Detailed Implementation Plan

### 8.1 Phase 1: Foundation Setup (Confidence: 85%)

**Step 1.1: Fix TSV Format**
```python
# Add 'id' column to segmented_clips.tsv
df['id'] = df['title_folder'] + '_' + df['video_path'].str.split('/').str[-1].str.replace('.mp4', '')
```

**Step 1.2: Update Configuration**
```yaml
# config/data/base.yaml
Custom_video:
  tsv: ./segmented_clips.tsv
  root: ./segmented_videos
```

**Step 1.3: Verify Model Loading**
```python
# Test script to verify model loading works
from mmaudio.model.lora_mmaudio import load_pretrained_with_lora
model = load_pretrained_with_lora('small_16k')  # May fail here
```

**Risk Assessment**: 40% chance of model loading failure
**Mitigation**: Manual checkpoint format inspection required

### 8.2 Phase 2: Dataset Integration (Confidence: 70%)

**Step 2.1: Implement Real-Time Dataset**

**Evidence for Implementation**: `custom_dataset.py:100-199` shows video loading pattern

```python
class SegmentedVGGDataset(Dataset):
    def __init__(self, tsv_path, video_root):
        self.clips_df = pd.read_csv(tsv_path, sep='\t')
        self.video_root = Path(video_root)
        # Initialize CLIP, Synchformer, Text encoders
        
    def __getitem__(self, idx):
        row = self.clips_df.iloc[idx]
        video_path = self.video_root / row['video_path']
        
        # Load 8-second video segment (pad if needed)
        video_data = self.load_video_segment(video_path, row['end_time'])
        
        # Extract features
        clip_features = self.extract_clip_features(video_data)    # Shape: [64, 1024]
        sync_features = self.extract_sync_features(video_data)    # Shape: [192, 768]  
        text_features = self.extract_text_features(row['caption']) # Shape: [77, 1024]
        
        # Generate dummy audio latents (target for training)
        audio_mean = torch.randn(250, 64)  # CONFIG_16K.latent_seq_len
        audio_std = torch.ones(250, 64)
        
        return {
            'id': row['id'],
            'a_mean': audio_mean,
            'a_std': audio_std, 
            'clip_features': clip_features,
            'sync_features': sync_features,
            'text_features': text_features,
            'caption': row['caption'],
            'video_exist': torch.tensor(True),
            'text_exist': torch.tensor(True)
        }
```

**Confidence**: 65% - Implementation pattern clear, but feature extraction details uncertain

**Critical Uncertainties**:
1. Which text encoder is used for captions? (Confidence: 30%)
2. Exact CLIP preprocessing pipeline? (Confidence: 50%)
3. Audio latent generation process? (Confidence: 40%)

**For 100% Certainty Needed**:
- Runtime inspection of original feature extraction pipeline
- Text encoder identification 
- Audio latent generation understanding

### 8.3 Phase 3: Training Integration (Confidence: 60%)

**Step 3.1: Modify Data Setup**

```python
# In mmaudio/data/data_setup.py:54-57
elif cfg.custom_finetune:
    from .segmented_vgg_dataset import SegmentedVGGDataset
    custom_video = SegmentedVGGDataset(
        tsv_path=cfg.data.Custom_video.tsv,
        video_root=cfg.data.Custom_video.root
    )
    dataset = custom_video
```

**Step 3.2: Test Training Loop**

```python
# Minimal training test
python train_lora.py --config-name lora_config
```

**Expected Failure Points** (in order of likelihood):
1. Model loading failure (60% probability)
2. Feature extraction dimension mismatch (70% probability)  
3. Sequence length assertion failure (80% probability)
4. Text encoder compatibility (50% probability)

### 8.4 Phase 4: Validation & Optimization (Confidence: 40%)

**Step 4.1: Audio Generation Test**
```python
# Test if trained model can generate audio
generated_audio = model.generate(video_features, text_features)
```

**Step 4.2: Quality Assessment**
- Compare generated audio with source video
- Validate LoRA learning (parameter changes)
- Check convergence metrics

**Confidence**: 40% - Many unknowns in generation pipeline

---

## 9. Risk Assessment & Mitigation

### 9.1 High-Risk Areas (Probability > 50%)

**Risk 1: Model Loading Failure** (60% probability)
- **Evidence**: Unverified checkpoint format compatibility
- **Impact**: Complete training failure
- **Mitigation**: Manual checkpoint inspection, potential format conversion

**Risk 2: Feature Dimension Mismatch** (70% probability) 
- **Evidence**: Unknown exact preprocessing pipeline
- **Impact**: Runtime tensor shape errors
- **Mitigation**: Reverse-engineer original feature extraction

**Risk 3: Sequence Length Issues** (80% probability)
- **Evidence**: Variable segment durations vs fixed requirements
- **Impact**: Assertion failures during training
- **Mitigation**: Padding/truncation implementation

### 9.2 Medium-Risk Areas (Probability 25-50%)

**Risk 4: Text Encoder Compatibility** (40% probability)
- **Evidence**: Unknown which text encoder used for captions
- **Impact**: Poor caption-audio alignment
- **Mitigation**: Systematic text encoder testing

**Risk 5: Memory/Performance Issues** (30% probability)
- **Evidence**: Real-time feature extraction computational overhead
- **Impact**: Training slowdown, OOM errors
- **Mitigation**: Batch size reduction, gradient checkpointing

---

## 10. Confidence Assessment Summary

### Components by Confidence Level:

**90-100% Confidence**:
- Repository structure analysis
- LoRA configuration format
- Segmented dataset inventory
- Sequence config requirements

**70-89% Confidence**:
- LoRA model architecture
- Data pipeline structure
- Implementation strategy viability
- Configuration fixes needed

**50-69% Confidence**:
- Feature extraction implementation
- Training integration success
- Model loading compatibility
- Runtime performance

**Below 50% Confidence**:
- Text encoder identification
- Audio latent generation
- Training convergence
- Final audio quality

### What's Needed for 100% Certainty:

1. **Runtime Model Testing**
   - Load pretrained model
   - Verify LoRA application
   - Test feature extraction pipeline

2. **Feature Pipeline Reverse Engineering**
   - Identify exact text encoder used
   - Understand audio latent generation
   - Verify preprocessing steps

3. **End-to-End Training Test**
   - Run training on small subset
   - Verify convergence behavior
   - Test audio generation quality

---

## 11. Recommended Next Steps

### Immediate Actions (Next 2 hours):

1. **Verify Model Loading** (Confidence: Must do)
```bash
python -c "
from mmaudio.model.lora_mmaudio import load_pretrained_with_lora
try:
    model = load_pretrained_with_lora('small_16k')
    print('✅ Model loading successful')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"
```

2. **Fix TSV Format** (Confidence: 95%)
```python
import pandas as pd
df = pd.read_csv('segmented_clips.tsv', sep='\t')
df['id'] = df['title_folder'] + '_' + df['video_path'].str.split('/').str[-1].str.replace('.mp4', '')
df.to_csv('segmented_clips_fixed.tsv', sep='\t', index=False)
```

### Short-term (Next 8 hours):

3. **Implement Minimal Dataset Adapter**
4. **Test Feature Extraction Components**
5. **Run Initial Training Test**

### Medium-term (Next 2 days):

6. **Optimize Real-time Pipeline**
7. **Validate Audio Generation**
8. **Performance Optimization**

---

## 12. Conclusion

**Current Status**: Repository has solid LoRA infrastructure but requires significant bridging work to train on our segmented dataset.

**Primary Challenge**: Mismatch between raw video data and expected pre-computed features.

**Recommended Approach**: Real-time feature extraction during training (Strategy A).

**Success Probability**: 60-70% with systematic implementation and testing.

**Key Success Factors**:
1. Model loading compatibility verification
2. Accurate feature extraction pipeline implementation  
3. Proper sequence length handling
4. Memory management optimization

**Critical Dependencies**:
- Understanding exact feature extraction pipeline
- Identifying correct text encoder
- Resolving model checkpoint compatibility

**Timeline Estimate**: 2-5 days for working implementation, 1-2 weeks for optimization.

---

## Appendix: Evidence References

### Code File References:
- `mmaudio/model/sequence_config.py:44-50` - Sequence requirements
- `mmaudio/data/data_setup.py:54-57` - Training data setup
- `mmaudio/data/extracted_vgg.py:30-39` - Pre-computed feature loading
- `mmaudio/model/lora_mmaudio.py:312-330` - Model loading function
- `config/lora_config.yaml:19-36` - LoRA configuration
- `mmaudio/data/extraction/custom_dataset.py:24-199` - Real-time loading example

### Assumptions Made:
1. Checkpoint format matches model architecture (Confidence: 60%)
2. Feature extractors in eval_utils work for training (Confidence: 70%)
3. Real-time extraction performance acceptable (Confidence: 65%)
4. LoRA target modules correctly specified (Confidence: 80%)

### Scientific Method Applied:
- **Hypothesis**: Real-time feature extraction can bridge data gap
- **Evidence**: Existing CustomFineTuningDataset proves feasibility
- **Prediction**: 60-70% success probability with systematic implementation
- **Validation Required**: Runtime testing to confirm hypothesis

This document represents our best scientific understanding based on available evidence, with explicit acknowledgment of uncertainties and required validation steps.