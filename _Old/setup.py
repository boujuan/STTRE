from setuptools import setup, find_packages

# pip install -e .
# python -m STTRE.main

setup(
    name="STTRE",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'lightning',
        'numpy',
        'polars',
    ],
)
