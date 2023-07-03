import cv2 as cv

img = cv.imread('000007.jpg')
cv.imshow('Car', img)

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Car', gray)

# Blur an image: arg --> img, kernel_size
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
cv.imshow("Blur Car", blur)

# Edge Cascade: arg --> img, threshold1, threshold2
canny = cv.Canny(img, 125, 125)
cv.imshow("Canny Edge", canny)

# Dilating the image: increase the thickness of the foreground object in the image
dilated = cv.dilate(canny, (3,3), iterations=1)
cv.imshow("Dilated", dilated)

# Eroding: shrinks the boundaries of an object in an image
eroded = cv.erode(dilated, (3,3), iterations=3)
cv.imshow("Eroded", eroded)

# Resize: arg --> img, size
resized = cv.resize(img, (500,500))
cv.imshow("Resized", resized)

# Cropping
cropped = img[50:200, 200:400]
cv.imshow("Cropped", cropped)

cv.waitKey(0)