import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evallm",
    version="0.0.55",
    author="Anonymous Authors",
    description="Evaluate LLMs using DFAs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
        "openai==1.55.3",
        "httpx==0.25.2",
        "permacache==4.0.0",
        "matplotlib",
        "matplotlib-venn==1.1.1",
        "pandas",
        "scipy==1.14.0",
        "anthropic==0.37.1",
        "adjustText==1.3.0",
    ],
    extras_require={
        "visual": ["automata-lib[visual]==8.3.0"],
    },
)
