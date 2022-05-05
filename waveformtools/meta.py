import sys, os
import numpy as np


package_directory = os.path.dirname(os.path.abspath(__file__))

def get_version():
	''' Get the latest version number based on the last commit date. 

	'''

	print(package_directory)
	# Open the file
	with open(package_directory + '/date.txt', 'r') as vers_file:
		vers = vers_file.read()[:10]

	#print(vers)
	return vers
	

