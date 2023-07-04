import numpy as np
import cv2 as cv
import torch
from torchvision.transforms import ToTensor

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# Load the model
from resnet_model import ResNet50
model = ResNet50(image_channels=3, num_classes=5)
model.load_state_dict(torch.load("resnet50.pth", map_location=torch.device('cpu')))

# Test on the image
img = cv.imread(r'Faces\val\ben_afflek\3.jpg')
faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

for (x,y,w,h) in faces_rect:
    faces_roi = img[y:y+h, x:x+w] # Region of interest  
    faces_roi = cv.resize(faces_roi, (224,224), interpolation=cv.INTER_NEAREST) # Resize the image
    faces_roi = ToTensor()(faces_roi) # Convert to tensor
    faces_roi = faces_roi.unsqueeze(0) # Add a batch dimension

    # Pass the image to the model
    model.eval()
    with torch.no_grad():
        preds = model(faces_roi)

    _, predicted = torch.max(preds, dim=1)

    # Get the predicted class probabilities
    probabilities = torch.nn.functional.softmax(preds, dim=1)[0] * 100
    print(f'Label = {people[predicted]} with confidence {probabilities[predicted].item()}%')

    cv.putText(img, str(people[predicted]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=1)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2) 

cv.imshow('Detected Face', img)

cv.waitKey(0)


