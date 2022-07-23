import cv2
from cv2 import *
from random import randrange

face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Smile_Detector=cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector=cv2.CascadeClassifier('haarcascade_eye.xml')

cam = cv2.VideoCapture(0)

while True:
    
    frame_read,frame=cam.read()
    
    grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    face=face_detector.detectMultiScale(grayscale)
   
    
    for (x,y,w,h) in face:
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),3)
        
        the_face=frame[y:y+h, x:x+w]
        
        grayscale=cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)
    
        Smile = Smile_Detector.detectMultiScale(grayscale, scaleFactor=1.5, minNeighbors=20)
        
        eyes = eye_detector.detectMultiScale(grayscale,scaleFactor=1.3, minNeighbors=15)
        
        for (x_,y_,w_,h_) in Smile:

            cv2.rectangle(the_face,(x_,y_),(x_+w_,y_+h_),(randrange(0,256),randrange(0,256),randrange(0,256)),3)

        for (x_,y_,w_,h_) in eyes:

            cv2.rectangle(the_face,(x_,y_),(x_+w_,y_+h_),(randrange(0,256),randrange(0,256),randrange(0,256)),3)

            if len(Smile)>0:
                cv2.putText(frame,'Smile',(x,y+h+50),fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))

                cv2.putText(frame, 'Sad', (x, y + h + 50), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255, 255, 255))

            if len(eyes)>0:
                cv2.putText(frame,'identified',(x,y+h+100),fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))

    cv2.imshow('Smile_Detector',frame)
    key=cv2.waitKey(1)

    if key==113 or key==81:
        break

cam.release()

print("completed")