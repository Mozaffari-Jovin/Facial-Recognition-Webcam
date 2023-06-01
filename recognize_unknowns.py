import face_recognition
import cv2
import os
import pickle


#### Unpickling: converting the byte stream into the original structure (lists and dictionary etc.) 
with open('known_faces.pickle', 'rb') as f: # rb stands for read byte
	names, encodings = pickle.load(f)

# print(names)
# quit()

cap = cv2.VideoCapture(0) # from webcam

stop_program = False

while True:
	ret, frame = cap.read()

	if ret:

		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		locs = face_recognition.face_locations(frame_rgb, model='hog') # hog or cnn
		encods = face_recognition.face_encodings(frame_rgb, locs)

		for (loc, encod) in zip(locs, encods):
			
			top_left = loc[3], loc[0]
			bottom_right = loc[1], loc[2]

			results = face_recognition.compare_faces(encodings, encod, 0.5) # compare encodings with encod
			# print(results) # [False, False, True, True]
			# quit()
			cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

			for (i, res) in enumerate(results):
				if res:
					new_name = names[i]
					cv2.putText(frame, new_name, top_left, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
					break

		cv2.imshow("Security camera", frame)
		
		q = cv2.waitKey(1) # frame rate
		
		if (q == ord('q')) or (q == ord('Q')):
			stop_program = True
		
		if stop_program:
			break

	if stop_program:
		break

cap.release()
cv2.destroyWindow("Security camera")