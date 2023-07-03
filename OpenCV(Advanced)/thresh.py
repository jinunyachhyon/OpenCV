import cv2 as cv

img = cv.imread('000000000030.jpg')
cv.imshow('Vase', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Simple Thresholding
# threshold: arg--> [image], [threshold value], [max value], [threshold type]
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY) # if pixel value is greater than 150, then it will be 255, else 0
cv.imshow('Simple Thresholded', thresh)

threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV) # if pixel value is greater than 150, then it will be 0, else 255
cv.imshow('Simple Thresholded Inverse', thresh_inv)


# Adaptive Thresholding
# Adaptive Thresholding: arg--> [image], [max value], [adaptive method], [threshold type], [block size], [constant that is subtracted from mean]
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3) # Computes mean for kernel size of 11
cv.imshow('Adaptive Thresholding', adaptive_thresh)

adaptive_thresh_gaussian = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3) # Computes mean for kernel size of 11
cv.imshow('Adaptive Thresholding Gaussian', adaptive_thresh_gaussian)

cv.waitKey(0)