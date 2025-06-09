#!/usr/bin/env python3
"""
Test script to verify fine-tuning setup is working correctly.
Run this before starting the actual fine-tuning pipeline.
"""

import sys
import traceback
from pathlib import Path
import torch
import logging
import importlib.util

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported"""
    log.info("Testing imports...")
    
    required_modules = [
        'torch',
        'torchaudio', 
        'torchvision',
        'tensordict',
        'av',
        'open_clip',
        'hydra',
        'omegaconf',
        'pandas',
        'numpy',
        'PIL',
        'tqdm'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            log.info(f"✓ {module}")
        except ImportError as e:
            log.error(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        log.error(f"Failed to import: {failed_imports}")
        return False
    
    log.info("All imports successful")
    return True


def test_custom_modules():
    """Test that our custom modules can be imported"""
    log.info("Testing custom modules...")
    
    try:
        from mmaudio.model.finetune_wrapper import MemoryEfficientMMAudio
        log.info("✓ MemoryEfficientMMAudio")
    except Exception as e:
        log.error(f"✗ MemoryEfficientMMAudio: {e}")
        return False
    
    try:
        from mmaudio.data.data_setup import setup_training_datasets
        log.info("✓ setup_training_datasets (modified)")
    except Exception as e:
        log.error(f"✗ setup_training_datasets: {e}")
        return False
    
    log.info("Custom modules working")
    return True


def test_gpu_setup():
    """Test GPU availability and memory"""
    log.info("Testing GPU setup...")
    
    if not torch.cuda.is_available():
        log.error("✗ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    log.info(f"✓ {gpu_count} GPU(s) available")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        log.info(f"  GPU {i}: {props.name}, {memory_gb:.1f}GB")
        
        if memory_gb < 50:
            log.warning(f"  GPU {i} has limited memory ({memory_gb:.1f}GB < 50GB recommended)")
    
    # Test memory allocation
    try:
        test_tensor = torch.zeros(1000, 1000, device='cuda')
        del test_tensor
        torch.cuda.empty_cache()
        log.info("✓ GPU memory allocation test passed")
    except Exception as e:
        log.error(f"✗ GPU memory test failed: {e}")
        return False
    
    return True


def test_model_loading():
    """Test that we can load the base model"""
    log.info("Testing model loading...")
    
    try:
        from mmaudio.model.networks import get_my_mmaudio
        from mmaudio.eval_utils import all_model_cfg
        
        # Test model config
        model_name = 'small_16k'
        if model_name not in all_model_cfg:
            log.error(f"✗ Model config {model_name} not found")
            return False
        
        log.info(f"✓ Model config {model_name} found")
        
        # Test model creation (without loading weights)
        model = get_my_mmaudio(model_name)
        log.info(f"✓ Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        
        # Test memory-efficient wrapper
        from mmaudio.model.finetune_wrapper import MemoryEfficientMMAudio
        wrapped_model = MemoryEfficientMMAudio(model, freeze_ratio=0.5)
        
        trainable = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in wrapped_model.parameters())
        
        log.info(f"✓ Wrapped model: {trainable/1e6:.1f}M/{total/1e6:.1f}M trainable ({100*trainable/total:.1f}%)")
        
        return True
        
    except Exception as e:
        log.error(f"✗ Model loading failed: {e}")
        traceback.print_exc()
        return False


def test_config_files():
    """Test that configuration files exist and are valid"""
    log.info("Testing configuration files...")
    
    config_files = [
        'config/base_config.yaml',
        'config/finetune_config.yaml',
        'config/data/base.yaml'
    ]
    
    for config_file in config_files:
        config_path = Path(config_file)
        if not config_path.exists():
            log.error(f"✗ Config file missing: {config_file}")
            return False
        log.info(f"✓ {config_file}")
    
    # Test config loading
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load('config/finetune_config.yaml')
        log.info("✓ Configuration loading successful")
        
        # Check key settings
        required_keys = ['model', 'custom_finetune', 'batch_size', 'learning_rate']
        for key in required_keys:
            if key not in cfg:
                log.error(f"✗ Missing config key: {key}")
                return False
        
        log.info("✓ Required configuration keys present")
        
    except Exception as e:
        log.error(f"✗ Config loading failed: {e}")
        return False
    
    return True


def test_feature_extraction():
    """Test feature extraction components"""
    log.info("Testing feature extraction setup...")
    
    try:
        from mmaudio.model.utils.features_utils import FeaturesUtils
        
        # Test without loading weights (just structure)
        log.info("✓ FeaturesUtils import successful")
        
        # Test video reading
        import av
        log.info("✓ PyAV available for video reading")
        
        # Test CLIP
        import open_clip
        log.info("✓ OpenCLIP available")
        
        return True
        
    except Exception as e:
        log.error(f"✗ Feature extraction test failed: {e}")
        return False


def test_data_pipeline():
    """Test data pipeline components"""
    log.info("Testing data pipeline...")
    
    try:
        # Test video splitting
        from training.split_videos_for_finetuning import get_video_info
        log.info("✓ Video splitting module loaded")
        
        # Test dataset classes
        from mmaudio.data.extracted_vgg import ExtractedVGG
        log.info("✓ Dataset classes available")
        
        # Test TensorDict
        import tensordict
        log.info("✓ TensorDict available")
        
        return True
        
    except Exception as e:
        log.error(f"✗ Data pipeline test failed: {e}")
        return False


def test_training_components():
    """Test training-specific components"""
    log.info("Testing training components...")
    
    try:
        # Test flow matching
        from mmaudio.model.flow_matching import FlowMatching
        fm = FlowMatching()
        log.info("✓ FlowMatching initialized")
        
        # Test optimizer components
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        log.info("✓ Mixed precision components available")
        
        # Test distributed training
        import torch.distributed as dist
        log.info("✓ Distributed training components available")
        
        return True
        
    except Exception as e:
        log.error(f"✗ Training components test failed: {e}")
        return False


def test_system_resources():
    """Test system resources"""
    log.info("Testing system resources...")
    
    import psutil
    
    # Check RAM
    ram_gb = psutil.virtual_memory().total / 1e9
    log.info(f"System RAM: {ram_gb:.1f}GB")
    
    if ram_gb < 80:
        log.warning(f"Limited RAM ({ram_gb:.1f}GB < 100GB recommended)")
    else:
        log.info("✓ RAM sufficient")
    
    # Check disk space
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free / 1e9
    log.info(f"Free disk space: {free_gb:.1f}GB")
    
    if free_gb < 50:
        log.warning(f"Limited disk space ({free_gb:.1f}GB < 50GB recommended)")
    else:
        log.info("✓ Disk space sufficient")
    
    return True


def create_test_video():
    """Create a small test video for pipeline testing"""
    log.info("Creating test video...")
    
    try:
        import cv2
        import numpy as np
        import tempfile
        
        # Create a simple test video
        test_dir = Path('./test_data')
        test_dir.mkdir(exist_ok=True)
        
        video_path = test_dir / 'test_video.mp4'
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 25.0, (224, 224))
        
        # Generate 200 frames (8 seconds at 25fps)
        for i in range(200):
            # Create a simple animated frame
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Add some movement
            x = int(112 + 50 * np.sin(i * 0.1))
            y = int(112 + 30 * np.cos(i * 0.15))
            
            cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
            cv2.putText(frame, f'Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        
        # Add audio using ffmpeg
        import subprocess
        
        # Generate a simple sine wave audio
        audio_video_path = test_dir / 'test_video_with_audio.mp4'
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=8',
            '-c:v', 'copy', '-c:a', 'aac',
            '-shortest', str(audio_video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            log.info(f"✓ Test video created: {audio_video_path}")
            return str(audio_video_path)
        else:
            log.warning(f"FFmpeg failed: {result.stderr}")
            return str(video_path)  # Return video without audio
            
    except Exception as e:
        log.error(f"✗ Test video creation failed: {e}")
        return None


def run_mini_pipeline_test():
    """Run a mini version of the pipeline to test integration"""
    log.info("Running mini pipeline test...")
    
    # Create test video
    test_video = create_test_video()
    if not test_video:
        return False
    
    try:
        # Test video splitting
        from training.split_videos_for_finetuning import get_video_info
        
        video_info = get_video_info(test_video)
        if video_info:
            log.info(f"✓ Video info: {video_info['duration']:.1f}s, {video_info['fps']:.1f}fps")
        else:
            log.error("✗ Failed to get video info")
            return False
        
        # Test that we can create a minimal dataset config
        test_config = {
            'model': 'small_16k',
            'custom_finetune': True,
            'batch_size': 1,
            'data_dim': {
                'latent_seq_len': 250,
                'clip_seq_len': 64,
                'sync_seq_len': 192,
                'text_seq_len': 77,
                'clip_dim': 1024,
                'sync_dim': 768,
                'text_dim': 1024
            }
        }
        
        log.info("✓ Mini pipeline test successful")
        return True
        
    except Exception as e:
        log.error(f"✗ Mini pipeline test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("MMAudio Fine-tuning Setup Test")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Custom Modules", test_custom_modules),
        ("GPU Setup", test_gpu_setup),
        ("Model Loading", test_model_loading),
        ("Configuration Files", test_config_files),
        ("Feature Extraction", test_feature_extraction),
        ("Data Pipeline", test_data_pipeline),
        ("Training Components", test_training_components),
        ("System Resources", test_system_resources),
        ("Mini Pipeline", run_mini_pipeline_test)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Ready for fine-tuning.")
        print("\nNext steps:")
        print("1. Place your 20 videos in a directory")
        print("2. Run: ./run_finetuning_pipeline.sh /path/to/your/videos")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before proceeding.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -e .")
        print("- Check GPU drivers: nvidia-smi")
        print("- Verify config files exist")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)