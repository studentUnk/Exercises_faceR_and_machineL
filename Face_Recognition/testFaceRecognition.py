import face_recognition

# Load the jpg files into numpy arrays
duque_image = face_recognition.load_image_file("duque.jpeg")
duque_image1 = face_recognition.load_image_file("duque1.jpg")
duque_image2 = face_recognition.load_image_file("duque2.jpg")
trump_image = face_recognition.load_image_file("trump.jpeg")
trump_image1 = face_recognition.load_image_file("trump1.jpeg")
trump_image2 = face_recognition.load_image_file("trump2.jpg")

unk_image = face_recognition.load_image_file("unk-du.png")

# face encondings
try:
	duque_face_encoding = face_recognition.face_encodings(duque_image)[0]
	duque_face_encoding1 = face_recognition.face_encodings(duque_image1)[0]
	duque_face_encoding2 = face_recognition.face_encodings(duque_image2)[0]
	trump_face_encoding = face_recognition.face_encodings(trump_image)[0]
	trump_face_encoding1 = face_recognition.face_encodings(trump_image1)[0]
	trump_face_encoding2 = face_recognition.face_encodings(trump_image2)[0]
	unk_face_encoding = face_recognition.face_encodings(unk_image)[0]
except IndexError:
	print ("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
	quit()

known_faces = [
	duque_face_encoding,
	duque_face_encoding1,
	duque_face_encoding2,
	trump_face_encoding,
	trump_face_encoding1,
	trump_face_encoding2
]

# results is an array of true/false
results = face_recognition.compare_faces(known_faces, unk_face_encoding)

print("Is Duque? {}".format(results[0:3]))
print("Is Trump? {}".format(results[3:6]))
print("Unknown? {}".format(not True in results))
