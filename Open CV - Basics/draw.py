# DRAW SHAPES AND PUTTING TEXT
import cv2 as cv
import numpy as np

# Create a blank img
blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('Blank', blank)

# 1. Paint the image a certain color
blank[:] = 0,255,0
cv.imshow('Green', blank)

# 2. Draw a rectangle: arg --> (img, origin, end-point, color, thickness)
cv.rectangle(blank, (0,0), (250,250), (0,0,255), thickness=2)
cv.imshow('Rectangle', blank)

# 3. Draw a circle: arg --> (img, midpoint, radius, color, thickness)
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (255,0,0), thickness=-1)
cv.imshow("Circle", blank)

# 4. Draw a line: arg --> img, pt1, pt2, color, thickness
cv.line(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (98,79, 45), thickness=3)
cv.imshow("Line", blank)

# 5. Write text on an image: arg --> img, str, origin, font, font_scale, color, thickness
cv.putText(blank, 'Hello World', (255,255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), thickness=2)
cv.imshow('Text', blank)

cv.waitKey(0)