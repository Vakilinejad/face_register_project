import cv2
import os
import face_recognition
import pickle
import numpy as np

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the folder already exists
if not os.path.exists('./Saved Images'):
    # If it doesn't exist, create the folder
    os.makedirs('./Saved Images')

frame_num = 0
# loading camera
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
    unknown_face_detected = False
else:
    rval = False

while rval :


    # Convert the image to grayscale (required for detection)
    # rgb_frame = frame[:, :, ::-1]
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image

    # Draw rectangles around the detected faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    print(len(face_encodings))
    for element in face_encodings:
        print(element)
    cv2.imshow("preview", frame)
    key = cv2.waitKey(1) & 0xFF  # Wait for a key press and get the key value
    if key == ord('s') or key == ord('S'):  # If 'S' is pressed
        cv2.imwrite('./Saved Images/frame' + f"{frame_num}" + '.jpg', frame)  # Save the frame as 'saved_frame.jpg'
        print("Frame saved.")
    elif key == 27:  # If 'ESC' is pressed
        break  # Exit the loop

    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")
