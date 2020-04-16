# code from https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
import face_recognition
import cv2
import numpy as np

# get a reference to the "video"
video_capture = cv2.VideoCapture("videoDuqueTrump.mp4") # 0 == webcam

number_images = 3
number_people = 4

names_people = ["Duque","Trump","Guaido","Maduro"]

# load a sample of pictures and learn how to recognize it

person_image = [
	[ 
	face_recognition.load_image_file("duque.jpeg"),
	face_recognition.load_image_file("duque1.jpg"),
	face_recognition.load_image_file("duque2.jpg")
	],
	[
	face_recognition.load_image_file("trump.jpeg"),
	face_recognition.load_image_file("trump1.jpeg"),
	face_recognition.load_image_file("trump2.jpg")
	],
	[
	face_recognition.load_image_file("guaido.jpg"),
	face_recognition.load_image_file("guaido1.jpg"),
	face_recognition.load_image_file("guaido2.jpg")
	],
	[
	face_recognition.load_image_file("maduro.jpeg"),
	face_recognition.load_image_file("maduro1.jpeg"),
	face_recognition.load_image_file("maduro2.jpg")
	]
]

person_face_encoding = []
for i in range(0,number_people):
	person_face_t = []
	for j in range(0,number_images):
		person_face_t.append( face_recognition.face_encodings(person_image[i][j], num_jitters=50)[0] )
	person_face_encoding.append(person_face_t)

known_face_encodings = []
known_face_names = []
for i in range(0,number_people):
	for j in range(0,number_images):
		known_face_encodings.append(person_face_encoding[i][j])
		known_face_names.append(names_people[i]) # set the name for each image
	
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
	# Grab a single frame of video
	ret, frame = video_capture.read()
	
	# Resize frame of video to 1/4 size for faster face recognition process
	small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
	# Convert the image from BGR color to RGB
	rgb_small_frame = small_frame[:,:,::-1]
	# Only process every other frame of video to save time
	if process_this_frame:
		# Find all the fraces and face encodings in current frame
		face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=3)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
		face_names = []
		for face_encoding in face_encodings:
			# See if the face is a match for a known face
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
			#print (matches)
			name = "Unknown"
			
			# If a match was found in known_face_encondigs, use the first one
			'''if True in matches:
			 	first_match_index = matches.index(True)
			 	#print (first_match_index)
			 	name = known_face_names[first_match_index]'''
			
			# Or instead use the known face with the smallest distance to the new face
			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = known_face_names[best_match_index]
			
			face_names.append(name)
	
	process_this_frame = not process_this_frame
	
	# Display the results
	for (top,right,bottom,left), name in zip(face_locations,face_names):
		# Scale back up face locations
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4
		
		if(name == "Unknown"):
			# Draw a box around the face
			cv2.rectangle(frame, (left,top),(right,bottom),(0,0,255),2)
			# Draw a label with a name below the face
			cv2.rectangle(frame, (left,bottom-15),(right,bottom),(0,0,255),cv2.FILLED)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame, name,(left+4,bottom-4),font,0.4,(255,255,255),1)
			
		else: # Known face rectangle green
			#print ("Did you see a green rectangle?")
			#Draw rectangle
			cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0),2)
			# Draw a label with a name below the face
			cv2.rectangle(frame, (left,bottom-15),(right,bottom),(0,255,0),cv2.FILLED)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame, name,(left+4,bottom-4),font,0.4,(255,255,255),1)
			
	# Display the resulting image
	cv2.imshow("Video", frame)
	# Hit "q" on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
