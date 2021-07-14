#Setup file

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="waveformtools", # Replace with your own username
    version="0.0.1",
    author="Vaishak Prasad",
    author_email="vaishak@iucaa.in",
    description="Functions for handling waveform data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/vaishakp/waveformtools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2 :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)

