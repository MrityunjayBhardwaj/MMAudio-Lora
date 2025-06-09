#!/usr/bin/env python3
"""
Split long videos into 8-second clips for MMAudio fine-tuning.
Supports overlapping windows for data augmentation.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import av
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_video_info(video_path):
    """Get video duration and metadata"""
    try:
        container = av.open(str(video_path))
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0] if container.streams.audio else None
        
        # Calculate duration
        if video_stream.duration:
            duration = float(video_stream.duration * video_stream.time_base)
        else:
            # Fallback: count frames
            duration = video_stream.frames / video_stream.average_rate
            
        info = {
            'duration': duration,
            'fps': float(video_stream.average_rate),
            'width': video_stream.width,
            'height': video_stream.height,
            'has_audio': audio_stream is not None
        }
        
        container.close()
        return info
        
    except Exception as e:
        log.error(f"Error processing {video_path}: {e}")
        return None


def split_video_to_clips(video_path, segment_duration=8.0, overlap_ratio=0.5, min_duration=2.0):
    """
    Split a video into fixed-duration clips with optional overlap.
    
    Args:
        video_path: Path to input video
        segment_duration: Duration of each clip in seconds (default: 8.0)
        overlap_ratio: Overlap between clips (0-1, default: 0.5 = 50% overlap)
        min_duration: Minimum clip duration to keep (default: 2.0 seconds)
    
    Returns:
        List of clip metadata dictionaries
    """
    video_info = get_video_info(video_path)
    if not video_info:
        return []
        
    if not video_info['has_audio']:
        log.warning(f"Video {video_path} has no audio track, skipping...")
        return []
        
    duration = video_info['duration']
    clips = []
    
    # Calculate stride based on overlap
    stride = segment_duration * (1 - overlap_ratio)
    
    # Generate clips
    current_time = 0
    clip_idx = 0
    
    while current_time < duration:
        end_time = min(current_time + segment_duration, duration)
        actual_duration = end_time - current_time
        
        # Only keep clips that meet minimum duration
        if actual_duration >= min_duration:
            clip_id = f"{video_path.stem}_clip_{clip_idx:04d}"
            
            clips.append({
                'id': clip_id,
                'video_path': str(video_path.absolute()),
                'video_name': video_path.stem,
                'start_time': current_time,
                'end_time': end_time,
                'duration': actual_duration,
                'fps': video_info['fps'],
                'width': video_info['width'],
                'height': video_info['height'],
                'caption': f"Audio from {video_path.stem} at {current_time:.1f}s"
            })
            
            clip_idx += 1
            
        current_time += stride
        
        # Stop if remaining duration is too short
        if duration - current_time < min_duration:
            break
            
    return clips


def process_video_directory(input_dir, output_tsv, segment_duration=8.0, 
                          overlap_ratio=0.5, video_extensions=None):
    """Process all videos in a directory"""
    
    if video_extensions is None:
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")
        
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))
        
    video_files = sorted(set(video_files))
    log.info(f"Found {len(video_files)} video files")
    
    if not video_files:
        raise ValueError(f"No video files found in {input_dir}")
        
    # Process each video
    all_clips = []
    for video_path in tqdm(video_files, desc="Processing videos"):
        clips = split_video_to_clips(
            video_path, 
            segment_duration=segment_duration,
            overlap_ratio=overlap_ratio
        )
        all_clips.extend(clips)
        log.info(f"{video_path.name}: {len(clips)} clips")
        
    # Create DataFrame and save
    df = pd.DataFrame(all_clips)
    output_path = Path(output_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, sep='\t', index=False)
    log.info(f"Saved {len(all_clips)} clips to {output_path}")
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total videos processed: {len(video_files)}")
    print(f"Total clips generated: {len(all_clips)}")
    print(f"Average clips per video: {len(all_clips) / len(video_files):.1f}")
    print(f"Total duration: {df['duration'].sum() / 60:.1f} minutes")
    
    # Group by video
    print("\n=== Clips per video ===")
    for video_name, group in df.groupby('video_name'):
        print(f"{video_name}: {len(group)} clips ({group['duration'].sum():.1f}s)")
        
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Split videos into clips for MMAudio fine-tuning'
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True,
        help='Directory containing input videos'
    )
    parser.add_argument(
        '--output_tsv', 
        type=str, 
        default='./custom_data/clips.tsv',
        help='Output TSV file path'
    )
    parser.add_argument(
        '--segment_duration', 
        type=float, 
        default=8.0,
        help='Duration of each clip in seconds (default: 8.0)'
    )
    parser.add_argument(
        '--overlap', 
        type=float, 
        default=0.5,
        help='Overlap ratio between clips (0-1, default: 0.5)'
    )
    parser.add_argument(
        '--min_duration', 
        type=float, 
        default=2.0,
        help='Minimum clip duration to keep (default: 2.0)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 <= args.overlap < 1:
        raise ValueError("Overlap must be between 0 and 1")
        
    # Process videos
    process_video_directory(
        args.input_dir,
        args.output_tsv,
        segment_duration=args.segment_duration,
        overlap_ratio=args.overlap
    )


if __name__ == '__main__':
    main()