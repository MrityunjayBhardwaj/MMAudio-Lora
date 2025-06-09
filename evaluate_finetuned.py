#!/usr/bin/env python3
"""
Evaluation script for fine-tuned MMAudio models.
Generates samples and computes metrics.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path
import torch
import torchaudio
from omegaconf import OmegaConfig
import pandas as pd
from tqdm import tqdm

from mmaudio.eval_utils import ModelConfig, generate, load_video, make_video
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import get_my_mmaudio
from mmaudio.model.finetune_wrapper import MemoryEfficientMMAudio
from mmaudio.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


class FineTunedModelEvaluator:
    """Evaluator for fine-tuned models"""
    
    def __init__(self, model_checkpoint: str, base_model_name: str = 'small_16k'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model_name = base_model_name
        self.checkpoint_path = model_checkpoint
        
        # Load configuration from checkpoint
        checkpoint = torch.load(model_checkpoint, map_location='cpu')
        self.config = checkpoint.get('config', {})
        
        # Setup models
        self.setup_models()
        
    def setup_models(self):
        """Initialize models and load fine-tuned weights"""
        
        log.info("Loading fine-tuned model...")
        
        # Get base model config
        from mmaudio.eval_utils import all_model_cfg
        if self.base_model_name not in all_model_cfg:
            raise ValueError(f"Unknown base model: {self.base_model_name}")
            
        self.model_config = all_model_cfg[self.base_model_name]
        
        # Load base model
        base_model = get_my_mmaudio(self.base_model_name).to(self.device).eval()
        
        # Load fine-tuned weights
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Wrap with memory-efficient wrapper (for consistency)
        self.model = MemoryEfficientMMAudio(
            base_model=base_model,
            freeze_ratio=checkpoint.get('freeze_ratio', 0.5),
            enable_checkpointing=False,  # Disable for inference
            enable_augmentation=False   # Disable for inference
        ).eval()
        
        # Setup flow matching
        self.fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=50)
        
        # Setup feature utils
        self.feature_utils = FeaturesUtils(
            tod_vae_ckpt=self.model_config.vae_path,
            synchformer_ckpt=self.model_config.synchformer_ckpt,
            enable_conditions=True,
            mode=self.model_config.mode,
            bigvgan_vocoder_ckpt=self.model_config.bigvgan_16k_path,
            need_vae_encoder=False
        ).to(self.device).eval()
        
        log.info("Models loaded successfully")
        
    @torch.inference_mode()
    def generate_from_video(
        self, 
        video_path: str, 
        prompt: str = "",
        duration: float = 8.0,
        cfg_strength: float = 3.0,
        num_steps: int = 50,
        seed: int = 42
    ):
        """Generate audio for a single video"""
        
        # Setup RNG
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        
        # Load video
        video_info = load_video(Path(video_path), duration)
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        actual_duration = video_info.duration_sec
        
        # Generate audio
        audio_latent = generate(
            clip_video=clip_frames.unsqueeze(0),
            sync_video=sync_frames.unsqueeze(0),
            text=[prompt] if prompt else [""],
            feature_utils=self.feature_utils,
            net=self.model.base_model,  # Use base model for generation
            fm=self.fm,
            rng=rng,
            cfg_strength=cfg_strength
        )
        
        # Decode to waveform
        waveform = self.feature_utils.decode(audio_latent)
        
        return waveform, actual_duration
        
    def evaluate_on_dataset(
        self,
        test_videos_dir: str,
        output_dir: str,
        prompts_file: str = None,
        num_samples: int = None
    ):
        """Evaluate on a dataset of videos"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find video files
        video_dir = Path(test_videos_dir)
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(video_dir.glob(f"*{ext}"))
            
        if num_samples:
            video_files = video_files[:num_samples]
            
        # Load prompts if provided
        prompts = {}
        if prompts_file and Path(prompts_file).exists():
            df = pd.read_csv(prompts_file)
            prompts = dict(zip(df['video_name'], df['prompt']))
            
        # Generate results
        results = []
        
        for video_file in tqdm(video_files, desc="Generating audio"):
            try:
                # Get prompt
                prompt = prompts.get(video_file.stem, "")
                
                # Generate audio
                waveform, duration = self.generate_from_video(
                    str(video_file),
                    prompt=prompt,
                    duration=8.0
                )
                
                # Save audio
                audio_output = output_path / f"{video_file.stem}_generated.wav"
                
                # Convert to CPU and correct format
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                    
                sample_rate = 16000 if self.model_config.mode == '16k' else 44100
                torchaudio.save(
                    str(audio_output),
                    waveform.cpu(),
                    sample_rate
                )
                
                # Create video with generated audio
                video_output = output_path / f"{video_file.stem}_with_audio.mp4"
                make_video(
                    video_file,
                    audio_output,
                    video_output,
                    duration_sec=duration
                )
                
                results.append({
                    'video_file': video_file.name,
                    'prompt': prompt,
                    'audio_output': audio_output.name,
                    'video_output': video_output.name,
                    'duration': duration,
                    'success': True
                })
                
            except Exception as e:
                log.error(f"Error processing {video_file}: {e}")
                results.append({
                    'video_file': video_file.name,
                    'prompt': prompts.get(video_file.stem, ""),
                    'audio_output': None,
                    'video_output': None,
                    'duration': None,
                    'success': False,
                    'error': str(e)
                })
                
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path / 'evaluation_results.csv', index=False)
        
        # Print summary
        successful = results_df['success'].sum()
        total = len(results_df)
        
        print(f"\n=== Evaluation Summary ===")
        print(f"Total videos processed: {total}")
        print(f"Successful generations: {successful}")
        print(f"Success rate: {100 * successful / total:.1f}%")
        print(f"Results saved to: {output_path}")
        
        return results_df
        
    def compare_models(
        self,
        original_model_name: str,
        test_videos: list,
        output_dir: str,
        prompts: dict = None
    ):
        """Compare fine-tuned model with original"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load original model
        from mmaudio.eval_utils import all_model_cfg
        original_config = all_model_cfg[original_model_name]
        original_config.download_if_needed()
        
        original_model = get_my_mmaudio(original_model_name).to(self.device).eval()
        original_model.load_weights(
            torch.load(original_config.model_path, map_location=self.device, weights_only=True)
        )
        
        # Setup original feature utils
        original_features = FeaturesUtils(
            tod_vae_ckpt=original_config.vae_path,
            synchformer_ckpt=original_config.synchformer_ckpt,
            enable_conditions=True,
            mode=original_config.mode,
            bigvgan_vocoder_ckpt=original_config.bigvgan_16k_path,
            need_vae_encoder=False
        ).to(self.device).eval()
        
        comparisons = []
        
        for video_path in tqdm(test_videos, desc="Comparing models"):
            video_name = Path(video_path).stem
            prompt = prompts.get(video_name, "") if prompts else ""
            
            # Generate with fine-tuned model
            try:
                finetuned_audio, duration = self.generate_from_video(
                    video_path, prompt=prompt, seed=42
                )
                
                # Save fine-tuned audio
                ft_audio_path = output_path / f"{video_name}_finetuned.wav"
                sample_rate = 16000 if self.model_config.mode == '16k' else 44100
                
                if finetuned_audio.dim() == 3:
                    finetuned_audio = finetuned_audio.squeeze(0)
                if finetuned_audio.dim() == 1:
                    finetuned_audio = finetuned_audio.unsqueeze(0)
                    
                torchaudio.save(str(ft_audio_path), finetuned_audio.cpu(), sample_rate)
                
            except Exception as e:
                log.error(f"Fine-tuned generation failed for {video_name}: {e}")
                continue
                
            # Generate with original model
            try:
                rng = torch.Generator(device=self.device)
                rng.manual_seed(42)  # Same seed for fair comparison
                
                video_info = load_video(Path(video_path), 8.0)
                
                original_audio = generate(
                    clip_video=video_info.clip_frames.unsqueeze(0),
                    sync_video=video_info.sync_frames.unsqueeze(0),
                    text=[prompt] if prompt else [""],
                    feature_utils=original_features,
                    net=original_model,
                    fm=self.fm,
                    rng=rng,
                    cfg_strength=3.0
                )
                
                original_waveform = original_features.decode(original_audio)
                
                # Save original audio
                orig_audio_path = output_path / f"{video_name}_original.wav"
                
                if original_waveform.dim() == 3:
                    original_waveform = original_waveform.squeeze(0)
                if original_waveform.dim() == 1:
                    original_waveform = original_waveform.unsqueeze(0)
                    
                torchaudio.save(str(orig_audio_path), original_waveform.cpu(), sample_rate)
                
                comparisons.append({
                    'video': video_name,
                    'prompt': prompt,
                    'finetuned_audio': ft_audio_path.name,
                    'original_audio': orig_audio_path.name,
                    'success': True
                })
                
            except Exception as e:
                log.error(f"Original generation failed for {video_name}: {e}")
                comparisons.append({
                    'video': video_name,
                    'prompt': prompt,
                    'finetuned_audio': ft_audio_path.name,
                    'original_audio': None,
                    'success': False
                })
                
        # Save comparison results
        comparison_df = pd.DataFrame(comparisons)
        comparison_df.to_csv(output_path / 'model_comparison.csv', index=False)
        
        return comparison_df


def main():
    parser = ArgumentParser(description='Evaluate fine-tuned MMAudio model')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to fine-tuned model checkpoint'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='small_16k',
        help='Base model name'
    )
    parser.add_argument(
        '--test_videos',
        type=str,
        required=True,
        help='Directory containing test videos'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation_output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--prompts_file',
        type=str,
        help='CSV file with video prompts'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        help='Limit number of test samples'
    )
    parser.add_argument(
        '--compare_original',
        action='store_true',
        help='Compare with original model'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize evaluator
    evaluator = FineTunedModelEvaluator(args.checkpoint, args.base_model)
    
    # Run evaluation
    results = evaluator.evaluate_on_dataset(
        args.test_videos,
        args.output_dir,
        args.prompts_file,
        args.num_samples
    )
    
    # Optional comparison with original
    if args.compare_original:
        test_videos = list(Path(args.test_videos).glob("*.mp4"))[:5]  # Limit for comparison
        
        prompts = {}
        if args.prompts_file:
            df = pd.read_csv(args.prompts_file)
            prompts = dict(zip(df['video_name'], df['prompt']))
            
        comparison = evaluator.compare_models(
            args.base_model,
            [str(v) for v in test_videos],
            str(Path(args.output_dir) / 'comparison'),
            prompts
        )
        
        print(f"\nModel comparison saved to: {Path(args.output_dir) / 'comparison'}")


if __name__ == "__main__":
    main()