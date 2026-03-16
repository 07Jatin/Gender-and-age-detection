import cv2
import numpy as np
image = cv2.imread('sample.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
print(f'Detected {len(faces)} faces in sample.jpg')
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('Sample', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('Test complete.')
