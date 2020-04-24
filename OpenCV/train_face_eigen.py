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

class TrainEigenFaces:
 def __init__(self):
  # path of the file to recognize faces
  cascPath = "haarcascadeData/haarcascade_frontalface_default.xml"
  # load file 
  self.face_cascade = cv2.CascadeClassifier(cascPath)
  if len(sys.argv) < 2:
   print "usage: set face-directory <base_path>"
   sys.exit(1)
  # dir to save the faces
  self.face_dir = sys.argv[1]
  # name of the face
  '''
  self.face_name = sys.argv[2]
  # create path of the file
  self.path = os.path.join(self.face_dir, self.face_name)
  if not os.path.isdir(self.path): # if doesn't exist the dir
   os.mkdir(self.path) # create directory
  '''
  # start eigen face recognizer
  #self.model = cv2.face.createEigenFaceRecognizer()
  self.model = cv2.face.EigenFaceRecognizer_create()
  self.count_captures = 0 # data for the capture video
  self.count_timer = 0 # data for the capture video

 def capture_training_images(self):
  video_capture = cv2.VideoCapture(0) # set camera
  while True:
   self.count_timer += 1
   ret, frame = video_capture.read() # read frame
   inImg = np.array(frame) # transform frame to array
   outImg = self.process_image(inImg)
   cv2.imShow('Video', outImg)
   
   # release the capture on pressing 'q'
   if cv2.waitKey(1) & 0xFF == ord('q'):
    video_capture.release() # free the capture
    cv2.destroyAllWindows() # delete memory and frames
    return
   
 def process_image(self, inImg):
  frame = cv2.flip(inImg, 1) # flip (turn) image over y-axis
  rezised_width, resized_height = (112,92) # parameters to rezise image
  if self.count_captures < NUM_TRAINING:
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray scale
   # resize image
   gray_resized = cv2.resize(gray, (gray.shape[1]/RESIZE_FACTOR, gray.shape[0]/RESIZE_FACTOR))
   # set values to identify faces
   faces = self.face_cascade.detectMultiScale(
   	gray_resized,
   	scaleFactor=1.1,
   	minNeighbors=5,
   	minSize=(30,30),
   	flags=cv2.cv.CV_HAAR_SCALE_IMAGE
   )
   if len(faces) > 0: # if found a face
    areas = []
    for (x,y,w,h) in faces:
     areas.append(w*h)
    max_area, idx = max([(val,idx) for idx,val in enumerate(areas)])
    face_sel = faces[idx]
    
    x = face_sel[0]*RESIZE_FACTOR
    y = face_sel[1]*RESIZE_FACTOR
    w = face_sel[2]*RESIZE_FACTOR
    h = face_sel[3]*RESIZE_FACTOR
    
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (resized_width, resized_height))
    img_no = sorted([int(fn[:fn.find('.')]) for fn in os.listdir(self.path) if fn[0]!='.']+[0])[-1]+1
    
    if self.count_timer%FREQ_DIV == 0:
     cv2.imwrite('%s/%s.png' % (self.path, img_no), face_resized)
     self.count_captures += 1 # increase image saved
     print "Captured image: ", self.count_captures
    
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3) # draw a rectangle
    cv2.putText(frame, self.face_name, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1,(0, 255,0)) # text for rectangle
   elif self.count_captures == NUM_TRAINING:
    print "Training data captured. Press 'q' to exit."
    self.count_captures += 1
   
   return frame

 def eigen_train_data(self):
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
  self.model.save('eigen_trained_data.xml')
  print "Training completed succesfully"
  return 
 
if __name__ == '__main__':
 trainer = TrainEigenFaces()
 #trainer.capture_training_images()
 trainer.eigen_train_data()
 print "Training face eigenfaces has finished"
