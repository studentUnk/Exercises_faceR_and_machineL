# Base of the code from
# https://github.com/tanay-bits/cvlib/blob/master/Face%20Recognition/face_recog_eigen.py

import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image

RESIZE_FACTOR = 4 # not used in the moment

class RecogFaces:

 def __init__(self): # first function to be executed
  cascPath = "haarcascadeData/haarcascade_frontalface_default.xml"
  self.face_cascade = cv2.CascadeClassifier(cascPath) # set haarcascade
  if len(sys.argv) < 5: # check that all the values have been set
   print "usage: set face-directory <base_path> eigen-face-model <model_path> image-video <source_path> type-of-algorithm 1=Eigen 2=Fisher 3=LBPH"
   sys.exit(1)
  self.face_dir = sys.argv[1] # set directory with the faces
  if(sys.argv[4] == "1"):
   self.model = cv2.face.EigenFaceRecognizer_create()
  elif (sys.argv[4] == "2"):
   self.model = cv2.face.FisherFaceRecognizer_create()
  elif (sys.argv[4] == "3"):
   #self.model = cv2.face.LBPH_create()
   self.model = cv2.face.LBPHFaceRecognizer_create()
  else:
   print "No recognizer algorithm has been set"
   sys.exit(1)
  #self.model = cv2.face.EigenFaceRecognizer_create() # create eigen face recognizer
  self.face_names = []
  
  self.count_faces = 0
  
 def load_trained_data(self): # load the dataset of the trained data (csv)
  names = {}
  key = 0
  for (subdirs, dirs, files) in os.walk(self.face_dir):
   for subdir in dirs:
    names[key] = subdir # set the name of the folders
    key += 1
   self.names = names
   self.model.read(sys.argv[2]) # load model
 
 def show_image(self): # recognize in image
  img = cv2.imread(sys.argv[3]) # load face ro recognize
  #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  while True:
   inImg = np.array(img)
   #outImg, self.face_names = self.process_image_2(img)
   outImg, self.face_names = self.process_image_2(inImg)
   
   #cv2.imshow('Face Detection', outImg)
   # Matplot libraries to show the image, imshow doesn't work
   #Conversion from BGR to RGB
   plt.axis("off")
   plt.imshow(cv2.cvtColor(outImg, cv2.COLOR_BGR2RGB))
   plt.show()
   # No conversion from BGR to RGB
   #plt.imshow(outImg, cmap='gray', interpolation='bicubic')
   #plt.xticks([]), plt.yticks([])
   #plt.show()
   if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
  
 def show_video(self): # recognize in video
  #video_capture = cv2.VideoCapture(0) # set path of the video if it is a cam
  video_capture = cv2.VideoCapture(sys.argv[3])
  print sys.argv[3]
  while True:
   ret, frame = video_capture.read() # read video
   #print ret
   #print frame
   inImg = np.array(frame)
   #outImg, self.face_names = self.process_image(inImg)
   outImg, self.face_names = self.process_image_2(inImg)
   cv2.imshow('Video', outImg) # output of the video
   #cv2.imshow('Video',inImg)
   
   # When everything is done, release the capture on pressing 'q'
   if cv2.waitKey(1) & 0xFF == ord('q'):
    video_capture.release()
    cv2.destroyAllWindows()
    return
 
 def process_image_2(self, inImg): # process the image and find coincidence
  #frame = cv2.flip(inImg,1)
  frame = inImg
  #print frame
  #resized_width, resized_height = (112,92)
  resized_width, resized_height = (200,200)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #gray = cv2.cvtColor(np.float32(frame), cv2.COLOR_BGR2GRAY)
  #gray_resized = cv2.resize(gray, (gray.shape[1]/RESIZE_FACTOR, gray.shape[0]/RESIZE_FACTOR))
  faces = self.face_cascade.detectMultiScale(
   gray,
   #gray_resized,
   #scaleFactor=1.1,
   scaleFactor=1.3,
   #minNeighbors=5,
   minNeighbors=5
   #minSize=(30,30),
   #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
   #flags=cv2.CV_HAAR_SCALE_IMAGE
   )
  persons = []
  for i in range(len(faces)):
   #print "face " + str(i)
   face_i = faces[i]
   #x = face_i[0] * RESIZE_FACTOR
   #y = face_i[1] * RESIZE_FACTOR
   #w = face_i[2] * RESIZE_FACTOR
   #h = face_i[3] * RESIZE_FACTOR
   x = face_i[0]
   y = face_i[1]
   w = face_i[2]
   h = face_i[3]
   face = gray[y:y+h, x:x+w]
   face_resized = cv2.resize(face, (resized_width, resized_height))
   confidence = self.model.predict(face_resized) # Find coincidence
   valConfidence = 0
   # set the value of confidence
   # lower is "higher" the confidence
   if (sys.argv[4] == "1"):
    valConfidence = 3500 #eigen
   elif (sys.argv[4] == "2"):
    valConfidence = 50 #fisher
   else:
    valConfidence = 80 #lbph
   if confidence[1]<valConfidence:
    print "I know your face!"
    person = self.names[confidence[0]]
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.putText(frame, '%s - %.0f' % (person, confidence[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
    self.count_faces += 1
    cv2.imwrite("foundInTest/"+str(self.count_faces)+ "_" +self.names[confidence[0]]+".jpg",frame)
   else:
    print "unknown"
    person = 'unknown'
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2, )
    cv2.putText(frame, person, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2, cv2.LINE_AA)
   persons.append(person)
  return (frame, persons)
 
 def process_image(self, inImg): # process the image
  frame = cv2.flip(inImg,1)
  resized_width, resized_height = (112,92)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray_resized = cv2.resize(gray, (gray.shape[1]/RESIZE_FACTOR, gray.shape[0]/RESIZE_FACTOR))
  faces = self.face_cascade.detectMultiScale(
   gray_resized,
   scaleFactor=1.1,
   minNeighbors=5,
   minSize=(30,30),
   flags=cv2.cv.CV_HAAR_SCALE_IMAGE
   )
  persons = []
  for i in range(len(faces)):
   face_i = faces[i]
   x = face_i[0] * RESIZE_FACTOR
   y = face_i[1] * RESIZE_FACTOR
   w = face_i[2] * RESIZE_FACTOR
   h = face_i[3] * RESIZE_FACTOR
   face = gray[y:y+h, x:x+w]
   face_resized = cv2.resize(face, (resized_width, resized_height))
   confidence = self.model.predict(face_resized) # Find coincidence
   if confidence[1]<3500:
    print "I know your face!"
    person = self.names[confidence[0]]
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)
    cv2.putText(frame, '%s - %.0f' % (person, confidence[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
   else:
    print "unknown face"
    person = 'unknown'
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
    cv2.putText(frame, person, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
   persons.append(person)
  return (frame, persons)
 
if __name__ == '__main__':
 recognizer = RecogFaces()
 recognizer.load_trained_data()
 print "The data has been load"
 print "Press 'q' to quit"
 #recognizer.show_image()
 recognizer.show_video()
