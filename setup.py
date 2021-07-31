#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="hichip_object",
    version="0.10.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # scripts=["scripts/pipeline.py"],
    install_requires=[
        "pysam",
        "google-re2",
        "numpy",
        "scipy",
        "pandas",
        "numba",
        "ray",
        "statsmodels",
        "scikit-learn",
        "patsy",
        "networkx",
        "matplotlib",
        "rpy2",  # to use smooth.spline function
        "macs2",  # if needs to call peaks
    ],
)
