from setuptools import setup, find_packages

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
