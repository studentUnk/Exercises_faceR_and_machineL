# Base of the code from
# https://github.com/tanay-bits/cvlib/blob/master/Face%20Recognition/face_recog_eigen.py

import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt

RESIZE_FACTOR = 4

class RecogEigenFaces:

 def __init__(self): # First function to be executed when the variable is created
  cascPath = "haarcascadeData/haarcascade_frontalface_default.xml"
  self.face_cascade = cv2.CascadeClassifier(cascPath) # set haarcascade
  if len(sys.argv) < 4:
   print "usage: set face-directory <base_path> eigen-face-model <model_path> image-video <source_path>"
   sys.exit(1)
  self.face_dir = sys.argv[1] # set directory with the faces
  self.model = cv2.face.EigenFaceRecognizer_create() # create eigen face recognizer
  self.face_names = []
  
 def load_trained_data(self): # Load the csv of a trained dataset
  names = {}
  key = 0
  for (subdirs, dirs, files) in os.walk(self.face_dir):
   for subdir in dirs:
    names[key] = subdir # set the name of the folders
    key += 1
   self.names = names
   self.model.read(sys.argv[2]) # load model
 
 def show_image(self): # show image and the name if it is recognized
  img = cv2.imread(sys.argv[3]) # load face ro recognize
  #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  while True:
   inImg = np.array(img)
   #outImg, self.face_names = self.process_image_2(img)
   outImg, self.face_names = self.process_image_2(inImg)
   
   #cv2.imshow('Face Detection', outImg)
   # Matplot libraries to show the image, imshow doesn't work
   plt.imshow(outImg, cmap='gray', interpolation='bicubic')
   plt.xticks([]), plt.yticks([])
   plt.show()
   if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
  
 def show_video(self): # gets the path of the camera and shows a video to recognize face
  video_capture = cv2.VideoCapture(0) # set path of the video
  while True:
   ret, frame = video_capture.read() # read video
   inImg = np.array(frame)
   outImg, self.face_names = self.process_image(inImg)
   cv2.imshow('Video', outImg) # output of the video
   
   # When everything is done, release the capture on pressing 'q'
   if cv2.waitKey(1) & 0xFF == ord('q'):
    video_capture.release()
    cv2.destroyAllWindows()
    return
 
 def process_image_2(self, inImg): # Process the image to find a face
  #frame = cv2.flip(inImg,1)
  frame = inImg
  #resized_width, resized_height = (112,92)
  resized_width, resized_height = (200,200)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
   print "face " + str(i)
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
   if confidence[1]<3500:
    print "I know your face!"
    person = self.names[confidence[0]]
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)
    cv2.putText(frame, '%s - %.0f' % (person, confidence[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
   else:
    print "unknown"
    person = 'unknown'
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
    cv2.putText(frame, person, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
   persons.append(person)
  return (frame, persons)
 
 def process_image(self, inImg): # Process the image to find a face
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
    cv2.putText(frame, '%S - %.0f' % (person, confidence[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
   else:
    print "unknown face"
    person = 'unknown'
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
    cv2.putText(frame, person, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
   persons.append(person)
  return (frame, persons)
 
if __name__ == '__main__':
 recognizer = RecogEigenFaces()
 recognizer.load_trained_data()
 print "The data has been load"
 print "Press 'q' to quit"
 recognizer.show_image()
 #recognizer.show_video()
