# --------------------------------------
# The base of this code is from:
# https://github.com/tanay-bits/cvlib/blob/master/Face%20Recognition/face_train_eigen.py
# --------------------------------------

import numpy as np
import cv2
import sys
import os

FREQ_DIV = 5
RESIZE_FACTOR = 4
NUM_TRAINING = 100 # maximum amount of images

class TrainFaces:
 def __init__(self): # first function to be executed
  # path of the file to recognize faces
  cascPath = "haarcascadeData/haarcascade_frontalface_default.xml"
  # load file 
  self.face_cascade = cv2.CascadeClassifier(cascPath)
  if len(sys.argv) < 3:
   print "usage: set face-directory <base_path>"
   print "usage: set type-of-recognizer 1=Eigen 2=Fisher 3=LBPH"
   sys.exit(0)
  # dir to save the faces
  self.face_dir = sys.argv[1]
  # start eigen face recognizer
  #self.model = cv2.face.createEigenFaceRecognizer()
  if(sys.argv[2] == "1"):
   self.model = cv2.face.EigenFaceRecognizer_create()
  elif (sys.argv[2] == "2"):
   self.model = cv2.face.FisherFaceRecognizer_create()
  elif (sys.argv[2] == "3"):
   #self.model = cv2.face.LBPH_create()
   self.model = cv2.face.LBPHFaceRecognizer_create()
  else:
   print "No recognizer algorithm has been set"
   sys.exit(1)
  self.count_captures = 0 # data for the capture video
  self.count_timer = 0 # data for the capture video

 def train_data(self):
  imgs = []
  tags = []
  index = 0
  
  for (subdirs, dirs, files) in os.walk(self.face_dir):
   for subdir in dirs:
    img_path = os.path.join(self.face_dir, subdir)
    for fn in os.listdir(img_path):
     path = img_path + '/' + fn
     tag = index
     imgs.append(cv2.imread(path,0)) # add image
     tags.append(int(tag)) # add number
    index += 1 # increase index
  (imgs, tags) = [np.array(item) for item in [imgs, tags]]
  
  self.model.train(imgs,tags)
  checkDir = ""
  if (sys.argv[1][len(sys.argv[1])-1] == '/'):
   checkDir = sys.argv[1][0:len(sys.argv[1])-1]
  else:
   checkDir = sys.argv[1]
  if (sys.argv[2] == "1"):
   self.model.save(checkDir+'_eigen_trained_data.xml')
  elif (sys.argv[2] == "2"):
   self.model.save(checkDir+'_fisher_trained_data.xml')
  else:
   self.model.save(checkDir+'_lbph_trained_data.xml')
  print "Training completed succesfully"
  return 
 
if __name__ == '__main__':
 trainer = TrainFaces()
 trainer.train_data()
 print "Training faces has finished"
