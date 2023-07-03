import cv2 as cv
import numpy as np

img = cv.imread('000007.jpg')
cv.imshow("Car", img)

# Transformations = Translation, Rotation, Resizing, Flipping, Cropping

# 1. Translation = Shifting image
def translate(img, x, y):
    # Translation matrix 2x3:
    # 1st row [1,0,x]-->'x' specifies the amount of translation in the x-axis
    # 2nd row [0,1,y]-->'y' specifies the amount of translation in the y-axis
    transMat = np.float32([[1,0,x], [0,1,y]])  
    dimensions = (img.shape[1], img.shape[0]) # (width, height)
    return cv.warpAffine(img, transMat, dimensions) # Function of OpenCV lib. to perform translation

# -x --> left shift
# -y --> Up shift
# +x --> Right shift
# +y --> Down shift
translated = translate(img, 100, -50) # shift to right and up
cv.imshow('Translated', translated)


# 2. Rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2) # Sets rotation point to the center of the image
    
    # Rotation Mat: arg --> center, angle, scale
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

# +angle --> counter-clockwise
# -angle --> closkwise
rotated = rotate(img, -45)
cv.imshow("Rotated", rotated)


# 3. Resizing
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow("Resized", resized)


# 4. Flipping
# flipcode 0--> vertical flip
# flipcode 1--> horizontal flip
# flipcode -1--> both ways
flip = cv.flip(img, -1)
cv.imshow("Flipped", flip)


# 5. Cropping
cropped = img[200:300, 300:400]
cv.imshow("Cropped", cropped)

cv.waitKey(0)