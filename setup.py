"""
setup.py - 项目安装配置

安装方式:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path


def read_requirements():
    """读取 requirements.txt"""
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        with open(req_file, "r", encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    return []


setup(
    name="traffic-spatiotemporal-backtrack",
    version="1.0.0",
    description="文本驱动的交通目标检索与跨镜时空回溯系统",
    author="Traffic Risk Perception Team",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*"]),
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "traffic-server=scripts.run_server:main",
            "traffic-preprocess=scripts.preprocess_video:main",
            "traffic-build-index=scripts.build_index:main",
            "traffic-demo=scripts.demo_inference:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
