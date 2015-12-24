#!/usr/bin/python
import numpy as np
import cv2
import sys

OPENCV_DATA_DIR = '/Applications/opencv-2.4.9/data/'

face_cascade = cv2.CascadeClassifier(OPENCV_DATA_DIR + 'haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(OPENCV_DATA_DIR + 'haarcascades/haarcascade_eye.xml')
print face_cascade

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame by frame
    ret, frame = cap.read()

    # Operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(250,250),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    
    # Draw a rectangle around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            maxSize=(w/6,y/5),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )
        # Draw rectangles around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
