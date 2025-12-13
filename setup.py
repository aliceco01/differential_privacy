"""
Differential Privacy for Anomaly Detection

A research project implementing differential privacy techniques for
machine learning models, with focus on autoencoders and PCA.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = [
        "numpy>=1.19.5,<1.24.0",
        "tensorflow>=2.8.0,<2.13.0",
        "matplotlib>=3.3.0,<3.8.0",
    ]

setup(
    name="differential-privacy-ae",
    version="0.1.0",
    author="Research Team",
    description="Differential privacy techniques for autoencoders and anomaly detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/differential_privacy",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7,<3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
            "jupyter>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dp-train=train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
