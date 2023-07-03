"""
Image contouring is the process of identifying structural outlines of objects in an image, 
which can help identify the shape of the object.
"""

import cv2 as cv
import numpy as np

img = cv.imread('000179.jpg')
cv.imshow('Ship', img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

# GrayScale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# 1. Finding contours using Canny Edge Detection
# Canny Edge Detection: arg --> threshold1, threshold2
canny = cv.Canny(gray, 125, 175)
cv.imshow('Canny', canny)

# Contours: arg --> image, mode, method
# hierarchies--> the relationship between contours
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')


# 2. Finding contours using thresholding
# Thresholding: binarize the image
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

# Contours: arg --> image, mode, method
# hierarchies--> the relationship between contours
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

# Draw contours on blank image: arg --> image, contours, contour_index, color, thickness
# contour_index means how many contours to draw; -1: draw all contours
cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)