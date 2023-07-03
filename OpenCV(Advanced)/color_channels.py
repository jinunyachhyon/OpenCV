import cv2 as cv
import numpy as np

img = cv.imread('000179.jpg')
cv.imshow('Ship', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

# Split image into color channels
b,g,r = cv.split(img)

cv.imshow('Blue', b)
cv.imshow('Green', g)
cv.imshow('Red', r)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

# Merge color channels
merged = cv.merge([b,g,r])
cv.imshow('Merged', merged)

# Displaying individual color channels
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

cv.waitKey(0)