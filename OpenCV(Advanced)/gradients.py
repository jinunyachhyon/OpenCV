import cv2 as cv

img = cv.imread('000179.jpg')
cv.imshow('Ship', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacian Gradient
lap = cv.Laplacian(gray, cv.CV_64F) # cv.CV_64F is the data type of the output image
lap = cv.convertScaleAbs(lap) # convertScaleAbs() is used to convert the output image to uint8
cv.imshow('Laplacian', lap)

# Sobel Gradient: computes gradients in x and y direction
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0) # 1 is the order of the derivative in x direction
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1) # 1 is the order of the derivative in y direction
cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)

combined_sobel = cv.bitwise_or(sobelx, sobely)
cv.imshow('Combined Sobel', combined_sobel)

# Canny Edge Detector: more advanced than the above two
canny = cv.Canny(gray, 150, 175) # 150 and 175 are the threshold values
cv.imshow('Canny', canny)

cv.waitKey(0)