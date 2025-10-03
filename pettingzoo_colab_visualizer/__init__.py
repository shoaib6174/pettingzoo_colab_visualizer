"""
pettingzoo_colab_visualizer

Lightweight wrapper exposing the public API for the package.

Public:
    - save_gif(frames, episode_name, folder="recordings", fps=20)
    - create_video_from_gifs(gif_folder="recordings", output_file="training_summary.mp4", fps=20, resize_width=None)
"""

__all__ = ["save_gif", "create_video_from_gifs", "__version__"]

__version__ = "0.1.0"

from .recorder import save_gif, create_video_from_gifs
