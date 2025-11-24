"""
Setup script for Chess AI package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chess-ai-alphazero",
    version="1.0.0",
    author="Chess AI Team",
    description="A self-learning chess AI using AlphaZero algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chess-ai-alphazero",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "python-chess>=1.9.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "viz": ["svgwrite>=1.4.0", "matplotlib"],
    },
    entry_points={
        "console_scripts": [
            "chess-ai-train=train:main",
            "chess-ai-play=play:main",
            "chess-ai-generate-moves=generate_moves:main",
        ],
    },
)
