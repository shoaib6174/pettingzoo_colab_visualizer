"""pettingzoo_colab_visualizer.recorder


Two main functions:


- save_gif(frames, episode_name, folder="recordings", fps=20)
-> saves a GIF of `frames` to `folder/{episode_name}.gif`.


- create_video_from_gifs(gif_folder, output_file="training_summary.mp4", fps=20, resize_width=None)
-> reads all gifs in `gif_folder`, draws episode numbers onto frames using OpenCV,
converts each gif into a clip, concatenates clips, and writes a single mp4.


This module avoids ImageMagick by drawing text with OpenCV, so it works in Colab
without extra system deps.
"""


import os
import re
from pathlib import Path
from typing import List, Optional


import imageio
import numpy as np
import cv2
from moviepy.editor import ImageSequenceClip, concatenate_videoclips


def extract_episode_number(filename: str) -> int:
    """Extract integer episode number from filename like 'ep_42.gif'."""
    m = re.search(r"ep(\d+)", filename, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0  # fallback if no number found


def _safe_episode_number_from_filename(name: str) -> str:
    """Try to extract an integer episode number from a filename.


    Examples that work: 'pong_ep100.gif', 'ep42.gif', 'episode_7.gif'.
    If nothing found, returns the basename without extension.
    """
    base = Path(name).stem
    # common patterns
    m = re.search(r"ep(\d+)", base, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"(episode|ep|episo|e)_(\d+)", base, flags=re.IGNORECASE)
    if m:
        return m.groups()[-1]
    m = re.search(r"(\d{1,6})", base)
    if m:
        return m.group(1)
    return base



def _add_episode_text_to_frames(frames: List[np.ndarray], episode_text: str, font_scale: float = 1.0, thickness: int = 2) -> List[np.ndarray]:
    """Draw a semi-opaque banner and white text at the top-center of each frame using OpenCV.


    Returns a new list of frames (RGB uint8).
    """
    out = []
    for f in frames:
        # ensure uint8 numpy array
        img = f.copy()
        if img.dtype != np.uint8:
            img = (255 * np.clip(img, 0, 1)).astype(np.uint8)


        # OpenCV expects BGR, but we will keep RGB and use cv2 functions assuming RGB order
        # draw text in white with a black semi-transparent rectangle behind
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = episode_text
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (w - text_w) // 2
        y = text_h + 10


        # rectangle background (semi-opaque)
        rect_x1 = max(x - 8, 0)
        rect_y1 = max(y - text_h - 8, 0)
        rect_x2 = min(x + text_w + 8, w)
        rect_y2 = min(y + 8, h)


        # overlay rectangle using alpha blending
        overlay = img.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


        # put white text (RGB)
        cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


        out.append(img)
    return out


def save_gif(frames: List[np.ndarray], episode_name: str, folder: str = "recordings", fps: int = 20) -> str:
    """Save a list of frames as a GIF named `episode_name`.gif inside `folder`.


    Args:
    frames: list of HxWx3 uint8 or float arrays (0..1 or 0..255).
    episode_name: base name for gif (e.g. 'ep100' or 'episode_100').
    folder: destination folder.
    fps: frames per second for GIF.


    Returns the path to the saved gif file.
    """
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)


    filename = folder_path / f"{episode_name}.gif"


    # normalize frames to uint8
    safe_frames = []
    for f in frames:
        arr = np.array(f, copy=False)
        if arr.dtype != np.uint8:
            arr = (255 * np.clip(arr, 0, 1)).astype(np.uint8)
        safe_frames.append(arr)


    imageio.mimsave(str(filename), safe_frames, fps=fps)


    return str(filename)


def create_video_from_gifs(gif_folder: str = "recordings", output_file: str = "training_summary.mp4", fps: int = 20, resize_width: Optional[int] = None) -> str:
    """Read all GIFs from `gif_folder`, annotate them with episode numbers using OpenCV,
    and produce one concatenated MP4 saved to `output_file`.


    The function sorts gif files alphanumerically and attempts to extract episode numbers
    from filenames using common patterns like 'ep123'.


    Returns the path to the created MP4.
    """
    gif_dir = Path(gif_folder)
    if not gif_dir.exists():
        raise FileNotFoundError(f"gif_folder '{gif_folder}' does not exist")


    # gif_files = sorted([p for p in gif_dir.iterdir() if p.suffix.lower() in ('.gif',)], key=lambda p: p.name)
    
    
    gif_files = [p for p in gif_dir.iterdir() if p.suffix.lower() == ".gif"]
    gif_files.sort(key=lambda p: extract_episode_number(p.name))


    if not gif_files:
        raise FileNotFoundError(f"No gif files found in '{gif_folder}'")


    clips = []
    for gif_path in gif_files:
        ep = _safe_episode_number_from_filename(gif_path.name)
        # read frames from gif
        frames = list(imageio.mimread(str(gif_path)))
        # add episode text onto frames
        frames_with_text = _add_episode_text_to_frames(frames, f"Episode {ep}")


        # create clip for these frames
        clip = ImageSequenceClip(frames_with_text, fps=fps)
        if resize_width is not None:
            clip = clip.resize(width=resize_width)
        clips.append(clip)


    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(output_file, codec="libx264", logger=None)


    return str(output_file)


if __name__ == "__main__":
    # simple self-test (does nothing heavy) -- left as an example
    print("pettingzoo_colab_visualizer module loaded. Use save_gif() and create_video_from_gifs().")