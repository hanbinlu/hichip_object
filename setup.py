#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="hichip_object",
    version="0.10.0",
    author="Hanbin Lu",
    author_email="lhb032@gmail.com",
    license="LICENSE",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    scripts=["scripts/hicpro_to_mvp.py", "scripts/hichip_callpeaks.py"],
    install_requires=[
        "pysam",
        "google-re2",
        "numpy",
        "scipy",
        "pandas",
        "numba",
        "ray[default]",
        "statsmodels",
        "scikit-learn",
        "patsy",
        "networkx",
        "matplotlib",
        "rpy2",  # to use smooth.spline function
        "macs2",  # if needs to call peaks
    ],
)
