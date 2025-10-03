from setuptools import setup, find_packages


setup(
name="pettingzoo_colab_visualizer",
version="0.1.0",
description="Utilities to save PettingZoo episode GIFs during training and combine them into a summary MP4",
long_description=open("README.md", "r").read(),
long_description_content_type="text/markdown",
author="Your Name",
packages=find_packages(exclude=("tests", "docs")),
include_package_data=True,
install_requires=[
"imageio>=2.9",
"moviepy>=1.0",
"opencv-python>=4.5",
"numpy",
],
python_requires=">=3.8",
)