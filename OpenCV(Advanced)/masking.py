import cv2 as cv
import numpy as np

img = cv.imread('000000000030.jpg')
cv.imshow('Vase', img)

# Blank Image: need to be same size as the image
blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('Blank Image', blank)

# Create a circle which acts as a mask
mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2 - 50), 150, 255, -1)
cv.imshow('Mask', mask)

# Masked Image
masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Image', masked)

cv.waitKey(0)