from setuptools import setup, find_packages

setup(
    name="sensortrack",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "plotly",
        "pandas",
    ],
) 