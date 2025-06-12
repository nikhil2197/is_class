"""Module to extract frames from video using ffmpeg."""
import os
import subprocess
from typing import List


def extract_frames(video_path: str, frames_dir: str, frame_interval: float) -> List[str]:
    """
    Extract frames from a video at a specified interval.

    :param video_path: Path to the input video file.
    :param frames_dir: Directory where extracted frames will be saved.
    :param frame_interval: Seconds between frames to extract.
    :return: Sorted list of frame file paths.
    """
    os.makedirs(frames_dir, exist_ok=True)
    fps = 1.0 / frame_interval
    output_pattern = os.path.join(frames_dir, "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-vsync", "vfr",
        output_pattern
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(
        [
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    return frames
