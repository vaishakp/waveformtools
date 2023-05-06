#!/usr/bin/env python

import os


source_directory = os.path.dirname(os.path.abspath(__file__))

print(source_directory)

def write_git_version():
    """Write the latest version number based on the last commit date
    to the public folder and to __init__.py file """

    # print(package_directory)
    # Open the file
    # Get the version
    vers = os.popen(f'git -C {source_directory} log -1 --date=short | grep Date').read()[8:-1]
    vers = vers.replace('-', '.')

    # Write to public/version
    with open(source_directory+'/public/version', 'w') as vers_file:
        vers_file.writelines([vers])

    # write to __init__.py
    ee = os.popen(f"sed -i '/__version__/c\__version__ = \"{vers}\"' {source_directory}/waveformtools/__init__.py")
    #with open(package_directory+'/waveformtools/__init__.py', 'r') as init_file:
    #with open(package_directory + "/../public/date.txt", "r") as vers_file:
        #vers = vers_file.read()[:10]

    ee = os.popen(f"sed -i '/version/c\__version__ = \"{vers}\"' {source_directory}/setup.py")

    return vers


write_git_version()

