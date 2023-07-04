import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = r'Faces/train' # Directory of the training images

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

features = [] # List of all the faces in the training set
labels = [] # List of all the labels associated with the faces
def create_train():
    """
    Loops over every folder in the base folder(training folder). 
    And inside that folder, loop over every image and grab that face in that image 
    and add it to the training set. 
    """
    for person in people:
        """Loop over every folder"""
        path = os.path.join(DIR, person) # Path to the folder of the person
        label = people.index(person) # Label of the person

        for img in os.listdir(path):
            """Loop over every image"""
            img_path = os.path.join(path, img) # Path to the image

            img_array = cv.imread(img_path) # Read the image
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                """Grab the face in the image"""
                faces_roi = gray[y:y+h, x:x+w] # Region of interest
                features.append(faces_roi)
                labels.append(label)

# Create the training set
create_train()

# Before training the recognizer, we need to convert the features and labels list into numpy array
features = np.array(features, dtype='object')
labels = np.array(labels)

# Instantiate Face Recognizer (Local Binary Patterns Histogram (LBPH) Face Recognizer)
face_recognizer = cv.face.LBPHFaceRecognizer_create() 

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features, labels)

# Save the trained model
face_recognizer.save('face_trained.yml')

# Save features and labels to a file
np.save('features.npy', features)
np.save('labels.npy', labels)



