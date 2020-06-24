#!/usr/bin/env python3
"""Metadata for package to allow installation with pip."""

import setuptools
import os
import subprocess

with open("README.md", "r") as fh:
    long_description = fh.read()

exec(open(os.path.join("stnet", "__version__.py")).read())

subprocess.run(['env', 'python3', os.path.join(os.path.dirname(__file__), 'refresh_version.py')])

setuptools.setup(
    name="stnet",
    description="Deep learning on histopathology images.",
    author="Bryan He",
    author_email="bryanhe@stanford.edu",
    version=__version__,
    url="https://github.com/bryanhe/ST-Net",
    packages=setuptools.find_packages(),
    tests_require=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ]
)

