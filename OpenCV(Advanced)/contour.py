import cv2 as cv

img = cv.imread('000179.jpg')
cv.imshow('Ship', img)

cv.waitKey(0)