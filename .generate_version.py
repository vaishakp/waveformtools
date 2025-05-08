#!/usr/bin/env python

import os
from datetime import date
import re
source_directory = os.path.dirname(os.path.abspath(__file__))

# print(source_directory)

def get_current_version():
    """ Get the current version """
    line_found=False
    with open(f"{source_directory}/waveformtools/__init__.py") as vf:
        while not line_found:
            one_line = vf.readline()
            if "__version__" in one_line:
                line_found=True
    
    current_version = re.search(r'\d+.\d+.\d+[.\d+]*', one_line).group(0)
    current_major_version = re.search(r'\d+.\d+.\d+', one_line).group(0)
    version_segments = current_version.split(".")

    print(f"Current version {current_version}")
    print(f"Current major version {current_major_version}")
    
    if len(version_segments)==3:
        current_minor_version=None
    elif len(version_segments)==4:
        current_minor_version=version_segments[-1]
    else:
        raise ValueError("Version not read properly")

    print(f"Current minor version {current_minor_version}")

    return current_version, current_major_version, current_minor_version


def write_git_version():
    """Write the latest version number based on the last commit date
    to the public folder and to __init__.py file"""

    # print(package_directory)
    # Open the file
    # Get the version
    # vers = os.popen(f'git -C {source_directory} log -1 --date=short | grep Date').read()[8:-1]
    vers = str(date.today())
    vers = vers.replace("-", ".")

    current_version, current_major_version, current_minor_version = get_current_version()
    if current_minor_version is not None:
        vers+=f".{int(current_minor_version)+1}"
    elif current_major_version==vers:
        vers+=f".1"
    

    # Write to public/version
    with open(source_directory + "/public/version", "w") as vers_file:
        vers_file.writelines([vers])

    # write to __init__.py
    ee = os.popen(
        f"sed -i '/__version__/c\\__version__ = \"{vers}\"' {source_directory}/waveformtools/__init__.py"
    )
    # with open(package_directory+'/waveformtools/__init__.py', 'r') as init_file:
    # with open(package_directory + "/../public/date.txt", "r") as vers_file:
    # vers = vers_file.read()[:10]

    ee = os.popen(
        f"sed -i '/version/c\\ \tversion=\"{vers}\",' {source_directory}/setup.py"
    )
    print("Version", vers)

    # Generate badge
    badge = f"""<svg width="140" height="20" xmlns="http://www.w3.org/2000/svg">
                <rect width="80" height="20" rx="0" ry="5" fill="grey" x="0" y="0" />
                <rect width="60" height="20" fill="blue" x="80" y="0" />
                <text x="6" y="13" fill="white" font-size="11">pypi package</text>
                <text x="83" y="14" fill="white" font-size="11">{vers}</text>
                </svg>
            """

    with open("docs/vers_badge.svg", "w") as svgf:
        svgf.write(badge)

    return vers


write_git_version()
