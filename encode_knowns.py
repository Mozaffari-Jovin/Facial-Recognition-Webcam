import face_recognition
import cv2
import os
import pickle # to save data

KNOWN_DIR = 'data/known_faces/'

known_names = os.listdir(KNOWN_DIR)

# print(known_names) # ['Jim Carrey', 'Nicolas Cage']

names =[]
encodings = []

#### KNOWN Figures
for name in known_names:
	# print(KNOWN_DIR+name) # data/known_faces/Jim Carrey
	# quit()
	file_names = os.listdir(KNOWN_DIR+name)
	# print(file_names) # ['J01.jpg', 'J01.jpg'] and ['N01.png', 'N02.jpg']
	for fn in file_names:
		# print(KNOWN_DIR+name+'/'+fn) # data/known_faces/Jim Carrey/J01.jpg
		# quit()
		img = cv2.imread(KNOWN_DIR+name+'/'+fn)
		rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		locs = face_recognition.face_locations(rgb, model='hog') # hog or cnn
		# print(locs) # [(81, 236, 236, 81)]
		# quit()
		encods = face_recognition.face_encodings(rgb, locs)
		# print(encods, encods[0].shape)
		# quit()
		names.append(name)
		encodings.append(encods[0])

# print(names, len(encodings))

#### Pickling Faces: converting the structure (lists and dictionary etc.) into a byte stream
with open('known_faces.pickle', 'wb') as f: # creating a file with the name and format of known_faces and wb (wb stands for write binary files)
	pickle.dump([names, encodings], f) # save names and emcodings in f

cv2.destroyAllWindows()