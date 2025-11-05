"""
GNN優秀人材分析システム - セットアップスクリプト
"""

from setuptools import setup, find_packages
from pathlib import Path

# プロジェクトディレクトリ
PROJECT_DIR = Path(__file__).parent

# README を読み込み
with open(PROJECT_DIR / "README.md", encoding="utf-8") as f:
    long_description = f.read()

# requirements.txt を読み込み
with open(PROJECT_DIR / "requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="talent-graph-analyzer",
    version="1.2.0",
    author="Skillnote",
    description="Graph Neural Network (GNN) を用いた優秀人材の特徴抽出・分析システム",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ykato27/talent-graph-analyzer",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: News/Diary",
    ],
    entry_points={
        "console_scripts": [
            "talent-analyzer=talent_analyzer.ui.app:main",
        ],
    },
    include_package_data=True,
    keywords="gnn graph-neural-network talent-analysis human-resources",
)
