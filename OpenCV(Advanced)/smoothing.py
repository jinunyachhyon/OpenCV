import cv2 as cv

img = cv.imread('000000000030.jpg')
cv.imshow('Vase', img)

# 1. Average Blur: Averages the pixels in the kernel area
average = cv.blur(img, (3,3)) # (3,3) is kernel size
cv.imshow('Average Blur', average)

# 2. Gaussian Blur: Gets weight form each area of the image and averages them
gauss = cv.GaussianBlur(img, (3,3), 0) # 0 is standard deviation
cv.imshow('Gaussian Blur', gauss)

# 3. Median Blur: Takes median of all the pixels under the kernel area
median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median)

# 4. Bilateral Blur: Most effective blur, retains edges
# sigma color: larger value means more colors in the neighborhood will be considered when blurring
# sigma space: larger value means pixels farther out from the center pixel will influence the blurring calculation
bilateral = cv.bilateralFilter(img, 10, 25, 35) # 10 is diameter, 25 is sigma color, 35 is sigma space
cv.imshow('Bilateral Blur', bilateral)

cv.waitKey(0)
