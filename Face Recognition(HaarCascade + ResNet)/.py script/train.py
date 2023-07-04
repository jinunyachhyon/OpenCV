import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = r'Faces\train' # Directory of the training images

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

dataset = [] # List of tuples (image, label)
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

            faces_rect = haar_cascade.detectMultiScale(img_array, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                """Grab the face in the image"""
                faces_roi = img_array[y:y+h, x:x+w] # Region of interest
                faces_roi = cv.resize(faces_roi, (224,224), interpolation=cv.INTER_NEAREST)
                dataset.append((faces_roi, label))

# Create the training set
create_train()

# Convert the images to tensors
dataset = [(ToTensor()(img), label) for img, label in dataset]

# Create the training loader
from torch.utils.data import DataLoader
train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

# Instantiate the model
from resnet_model import ResNet50
model = ResNet50(image_channels=3, num_classes=5).to(device)

# Loss and optimizer
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
from tqdm import tqdm
import time
num_epochs = 10
for epoch in tqdm(range(num_epochs), desc='Epochs'):

    model.train() # Set the model to training mode
    start_time = time.time() # Start time of the epoch

    running_loss = 0.0
    running_corrects = 0

    # Iterate over the training data in batches
    for inputs, labels in train_loader:  
        # Move the inputs and labels to the selected device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    end_time = time.time()  # End time of the epoch
    epoch_duration = end_time - start_time  # Duration of the epoch

    # Calculate epoch loss and accuracy for training data
    epoch_loss = running_loss / len(dataset)
    epoch_acc = running_corrects.double() / len(dataset)

    # Print the epoch duration
    tqdm.write(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")

    # Print the loss and accuracy 
    print(f"Epoch [{epoch+1}/{num_epochs}], "
        f"Overfit Loss: {epoch_loss:.4f}, Overfit Accuracy: {epoch_acc:.4f}")

# Save the model
torch.save(model.state_dict(), 'resnet50.pth')


