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
                #print eyes
                eyes = get_likeliest_eyes(eyes, face)
                if len(eyes) != 2:
                    continue
                if abs(np.log2(box_area(eyes[0])/box_area(eyes[1]))) > 1:
                    # area of eye boxes can't differ by more than factor of 2
                    continue
                for (ex, ey, ew, eh) in eyes:
                    # Draw rectangles around eyes
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
                    cv2.line(roi_color, (ex+ew/2,ey), (ex+ew/2,ey+eh), (255, 0, 0), 2)
                    eyes_midX = get_eyes_midpointX(eyes[0], eyes[1], face)
                    cv2.line(frame, (eyes_midX, y), (eyes_midX, y+h), (255, 0, 0), 2)
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.line(frame, (x+w/2,y), (x+w/2,y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def get_likeliest_eyes(eyes, face):
    face_centerX = face[0] + face[2]/2
    best_eyes = []
    best_dist = np.inf
    for i in xrange(len(eyes)):
        for j in xrange(i+1, len(eyes)):
            if ((face_centerX-get_eye_centerX(eyes[i], face))*
                (face_centerX-get_eye_centerX(eyes[j], face))) > 0:
                # centers of two eyes on same side of face
                continue
            eyes_midX = get_eyes_midpointX(eyes[i],eyes[j], face)
            new_dist = abs(face_centerX - eyes_midX)
            if best_dist > new_dist:
                best_eyes = [eyes[i],eyes[j]]
                best_dist = new_dist
    return best_eyes

def get_eye_centerX(eye, face):
    return face[0] + face[2]/8 + eye[0] + eye[2]/2
    
def get_eyes_midpointX(e1, e2, face):
    return (get_eye_centerX(e1, face) + get_eye_centerX(e2, face))/2

def box_area(box):
    return box[2]*box[3]

if __name__=="__main__":
    main()
