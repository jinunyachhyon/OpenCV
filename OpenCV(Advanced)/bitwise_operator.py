import cv2 as cv
import numpy as np

blank = np.zeros((400,400), dtype='uint8')

# Rectangle: arg --> (image, top-left corner, bottom-right corner, color, thickness)
rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1) # -1 is thickness, -1 fills the shape

# Circle: arg --> (image, center, radius, color, thickness)
circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

# 1. Bitwise AND: Intersecting regions
bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow('Bitwise AND', bitwise_and)

# 2. Bitwise OR: Non-intersecting and intersecting regions
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow('Bitwise OR', bitwise_or)

# 3. Bitwise XOR: Non-intersecting regions
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow('Bitwise XOR', bitwise_xor)

# 4. Bitwise NOT: Inverts binary color
bitwise_not = cv.bitwise_not(rectangle)
cv.imshow('Bitwise NOT', bitwise_not)

cv.waitKey(0)