from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="resonanceai",
    version="0.2.0",
    description="Hallucination detection via phoneme-resonance dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kriti",
    license="Apache 2.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "bert": ["torch>=2.0.0", "transformers>=4.30.0"],
        "benchmark": ["sentence-transformers>=2.2.0", "scikit-learn>=0.24.0"],
        "dev": ["pytest>=7.0.0", "hypothesis>=6.0.0"],
        "all": ["torch>=2.0.0", "transformers>=4.30.0",
                "sentence-transformers>=2.2.0", "scikit-learn>=0.24.0",
                "pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "resonanceai=urcm.cli:main",
        ],
    },
    include_package_data=True,
)
