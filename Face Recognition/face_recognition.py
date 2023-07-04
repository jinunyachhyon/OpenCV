import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
# Load features and labels
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

# Instantiate Face Recognizer (Local Binary Patterns Histogram (LBPH) Face Recognizer)
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Read the trained model
face_recognizer.read('face_trained.yml')

# Test on the image
img = cv.imread(r'Faces\val\ben_afflek\3.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Recognize the face in the image
# 1. Detect the face (Classify)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
# 2. Crop the face (Grab the face)
for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w] # Region of interest

    # 3. Pass the face to the recognizer (Predict)
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)

