import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evallm",
    version="0.0.55",
    author="Kavi Gupta, Armando Solar-Lezama",
    author_email="kavig+evallm@mit.edu",
    description="Evaluate LLMs using DFAs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kavigupta/evallm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy==2.0.0",
        "automata-lib==8.3.0",
        "openai==1.34.0",
        "permacache==3.8.0",
        "matplotlib",
        "pandas",
        "scipy==1.14.0",
    ],
    extras_require={
        "visual": ["automata-lib[visual]==8.3.0"],
    },
)
