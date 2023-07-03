# Histogram: Allows us to visualize the pixel intensity distribution of an image
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('000000000030.jpg')
cv.imshow('Vase', img)

blank = np.zeros(img.shape[:2], dtype='uint8')
# cv.imshow('Blank Image', blank)


# 1. For Grayscale Histogram
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2 - 50), 150, 255, -1)
# cv.imshow('Circle', circle)

masked_gray_image = cv.bitwise_and(gray, gray, mask=mask)
cv.imshow('Mask', mask)

# Grayscale Histogram: arg--> [image], [channels], mask, [histSize], [ranges]
gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])

plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()


# 2. For Color Histogram
mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2 - 50), 150, 255, -1)
# cv.imshow('Mask', mask)

masked_image = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Image', masked_image)

plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b','g','r')
for  i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()


cv.waitKey(0)