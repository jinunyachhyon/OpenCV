import cv2 as cv

img = cv.imread('group.jpg')
# img = cv.resize(img, (400, 600), interpolation=cv.INTER_AREA)
cv.imshow('Group Pic', img)

# Gray scale because uses edges to detect the face
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Haar Cascade Classifier to classifies either its a face or not
haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect the face
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3) # minNeighbors is the min no. of neighbors to detect the face
print(f'Number of faces found = {len(faces_rect)}')

# Draw rectangle around the face
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)
