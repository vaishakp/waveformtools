# Setup file

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


# Get version from init
import waveformtools
vers = waveformtools.get_version()


setuptools.setup(
    name="waveformtools",
    version=f"{vers}",
    author="Vaishak Prasad",
    author_email="vaishakprasad@gmail.com",
    description="Functions for handling waveform and numerical relativity data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/vaishakp/waveformtools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3",
    )

with open("pyproject.toml", 'r') as file:

    all_data = file.readlines()

for num, line in enumerate(all_data):
    if 'version' in line:
        all_data[num] = f"version = \"{vers}\"\n"
        break

with open("pyproject.toml", "w") as file:
    file.writelines(all_data)



