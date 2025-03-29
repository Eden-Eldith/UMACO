from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="umaco",
    version="0.1.0",
    author="Eden Eldith",
    author_email="pcobrien@hotmail.co.uk",
    description="Universal Multi-Agent Cognitive Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Eden-Eldith/UMACO",
    packages=find_packages(),
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
    }
)
