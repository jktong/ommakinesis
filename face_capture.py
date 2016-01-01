#!/usr/bin/python
import numpy as np
import cv2
import sys

OPENCV_DATA_DIR = '/Applications/opencv-2.4.9/data/'
HAARCASCADES_DIR = '%s/haarcascades/' % OPENCV_DATA_DIR

face_cascade = cv2.CascadeClassifier(HAARCASCADES_DIR + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(HAARCASCADES_DIR + 'haarcascade_eye.xml')

def main():
    cap = cv2.VideoCapture(0)

    while(True):    
        # Capture frame by frame
        ret, orig_frame = cap.read()    
        # Flip image around y-axis to mirror capture
        frame = cv2.flip(orig_frame,1)
        # Turn to greyscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,5)

        # Detect face
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(250,250),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )
        for (x, y, w, h) in faces:
            face = (x, y, w, h)
            # Restrict area to search for eyes significantly improves reliability
            roi_gray = gray[y+h/6:y+h/2, x+w/8:x+w*7/8]
            roi_color = frame[y+h/6:y+h/2, x+w/8:x+w*7/8]
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.09,
            minNeighbors=5,
            #maxSize=(w/6,y/5),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            if len(eyes) > 0:
                if len(eyes) > 2:
                    eyes = get_likeliest_eyes(eyes, face)
                for (ex, ey, ew, eh) in eyes:
                    # Draw rectangles around eyes
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def get_likeliest_eyes(eyes, face):
    face_centerX = face[0] + face[2]/2
    #best_eyes = np.empty((2,4))
    best_eyes = []
    best_dist = np.inf
    for i in xrange(len(eyes)):
        for j in xrange(i+1, len(eyes)):
            if ((face_centerX-(eyes[i][0]+eyes[i][2]/2))*
                (face_centerX-(eyes[j][0]+eyes[j][2]/2))) > 0:
                # centers of two eyes on same side of face
                continue
            eyes_midX = get_eyes_midpointX(eyes[i],eyes[j])
            new_dist = abs(face_centerX - eyes_midX)
            if best_dist > new_dist:
                best_eyes = [eyes[i],eyes[j]]
                best_dist = new_dist
    return best_eyes

def get_eye_centerX(eye):
    return eye[0] + eye[2]/2
    
def get_eyes_midpointX(e1, e2):
    return (get_eye_centerX(e1) + get_eye_centerX(e2))/2

if __name__=="__main__":
    main()
