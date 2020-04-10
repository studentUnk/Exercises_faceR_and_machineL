# Resources?
# https://www.zentut.com/python-tutorial/working-with-directory/
# https://stackoverflow.com/questions/8858008/how-to-move-a-file-in-python
# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory

#!/usr/bin/env python

import sys
import os.path
import shutil

if __name__ == "__main__":
	
	if len(sys.argv) != 2:
		print "usage: set the path to create folders <base_path>"
		sys.exit(1)
	
	BASE_PATH=sys.argv[1]
	nameFiles=[]
	nameDir=[]
	for (dirpath, dirnames, filenames) in os.walk(BASE_PATH):
		nameFiles.extend(filenames)
		#nameDir.extend(dirnames)
		break
	for n in nameFiles:
		findMe=False
		nameSubject=n[0:(n.find('.'))]
		#for d in nameDir:
		#	if d == nameSubject:
		#		findMe=True
		#		break
		if os.path.exists(BASE_PATH+nameSubject):
			if os.path.isdir(BASE_PATH+nameSubject):
				findMe=True
		if not findMe:
			os.mkdir(BASE_PATH+nameSubject)
		shutil.move(BASE_PATH+n,BASE_PATH+nameSubject)
		print nameSubject
