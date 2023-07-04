import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"

people = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling']
DIR = r'Faces\val' # Directory of the validation images

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

val_dataset = [] # List of tuples (image, label)
def create_val_dataset():
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

            faces_rect = haar_cascade.detectMultiScale(img_array, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                """Grab the face in the image"""
                faces_roi = img_array[y:y+h, x:x+w] # Region of interest
                faces_roi = cv.resize(faces_roi, (224,224), interpolation=cv.INTER_NEAREST)
                val_dataset.append((faces_roi, label))

# Create the training set
create_val_dataset()

# Convert the images to tensors
val_dataset = [(ToTensor()(img), label) for img, label in val_dataset]

# Create the training loader
from torch.utils.data import DataLoader
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False)

# Load the model
from resnet_model import ResNet50
model = ResNet50(image_channels=3, num_classes=5).to(device)
model.load_state_dict(torch.load("resnet50.pth", map_location=torch.device('cpu')))

# Testing the model
correct = 0
total = 0

with torch.no_grad(): # Disabling gradient calculation, since testing doesnot require weight update
    model.eval() # Set the model to evaluation mode

    for inputs, labels in test_loader:
      # Move the inputs and labels to the selected device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass 
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")