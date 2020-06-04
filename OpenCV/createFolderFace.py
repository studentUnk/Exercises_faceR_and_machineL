# Resources?
# https://www.zentut.com/python-tutorial/working-with-directory/
# https://stackoverflow.com/questions/8858008/how-to-move-a-file-in-python
# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory

#!/usr/bin/env python

import sys
import os.path
import shutil

if __name__ == "__main__":
	
	if len(sys.argv) != 2: # check that all the values exists
		print "usage: set the path to create folders <base_path>"
		sys.exit(1)
	
	BASE_PATH=sys.argv[1] # path to create folders
	nameFiles=[]
	nameDir=[]
	for (dirpath, dirnames, filenames) in os.walk(BASE_PATH):
		nameFiles.extend(filenames) # get all the name of the files in the folder
		#nameDir.extend(dirnames)
		break
	for n in nameFiles:
		findMe=False # there is no folder of the face
		nameSubject=n[0:(n.find('.'))] # get name of the subject (before '.')
		#for d in nameDir:
		#	if d == nameSubject:
		#		findMe=True
		#		break
		if os.path.exists(BASE_PATH+nameSubject): # check if the name exists
			if os.path.isdir(BASE_PATH+nameSubject): # check if it is a folder
				findMe=True # there is already a folder
		if not findMe: # there is no folder
			os.mkdir(BASE_PATH+nameSubject) # create folder
		shutil.move(BASE_PATH+n,BASE_PATH+nameSubject) # move image to the folder
		print nameSubject # print name of the image moved
