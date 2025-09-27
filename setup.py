from pathlib import Path
from typing import Dict

from setuptools import find_packages, setup


BASE_DIR = Path(__file__).parent


def read_long_description() -> str:
    with open(BASE_DIR / "README.md", "r", encoding="utf-8") as fh:
        return fh.read()


def read_version() -> str:
    version_ns: Dict[str, str] = {}
    init_path = BASE_DIR / "umaco" / "__init__.py"
    with open(init_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip().startswith("__version__"):
                exec(line, version_ns)
                break
    version = version_ns.get("__version__")
    if not version:
        raise RuntimeError("Unable to determine package version from umaco/__init__.py")
    return version


long_description = read_long_description()
version = read_version()

setup(
    name="umaco",
    version=version,
    author="Eden Eldith",
    author_email="pcobrien@hotmail.co.uk",
    description="Universal Multi-Agent Cognitive Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Eden-Eldith/UMACO",
    license="MIT",
    license_files=("LICENSE",),
    packages=find_packages(include=("umaco", "umaco.*")),
    py_modules=["Umaco13", "universal_solver", "umaco_gpu_utils"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "ripser>=0.6.0",
        "persim>=0.3.0",
        "scipy>=1.7.0",
        "cma>=3.1.0",
        "networkx>=2.6.0",
    ],
    extras_require={
        "llm": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "peft>=0.4.0", 
            "datasets>=2.10.0",
            "wandb>=0.15.0",
            "bitsandbytes>=0.41.0",
        ],
        "gpu": ["cupy-cuda11x>=11.0.0"],
    },
    project_urls={
        "Documentation": "https://github.com/Eden-Eldith/UMACO#readme",
        "Source": "https://github.com/Eden-Eldith/UMACO",
        "Tracker": "https://github.com/Eden-Eldith/UMACO/issues",
    },
)
