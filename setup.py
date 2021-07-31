#!/usr/bin/env python

from setuptools import setup

setup(
    name="hichip_object",
    version="0.10.0",
    packages=["hichip_object", "multipass_process"],
    scripts=["scripts/pipeline.py"],
    install_requires=[
        "pysam",
        "google-re2",
        "numpy",
        "scipy",
        "pandas",
        "numba",
        "ray",
        "statsmodels",
        "patsy",
        "networkx",
        "matplotlib",
        "rpy2",
    ],
)
