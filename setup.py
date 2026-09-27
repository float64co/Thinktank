from setuptools import setup, find_packages

setup(
    name="thinktank",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "anthropic>=0.50.0",
    ],
    entry_points={
        "console_scripts": [
            "thinktank=thinktank.tui:main",
        ],
    },
    python_requires=">=3.10",
)
