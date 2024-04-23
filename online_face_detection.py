import cv2
import os
import face_recognition

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
else:
    rval = False

while rval:

    # Convert the image to grayscale (required for detection)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image)
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

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
