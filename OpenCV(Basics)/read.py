import cv2 as cv

# Read the image 
img = cv.imread('000007.jpg')

# Display the read image 
# imshow --> 2 arg : name of img, the read img
cv.imshow('Car', img)

# Wait key --> waits for the keys to be pressed
cv.waitKey(0) # 0 means it waits for infinite time


# Reading videos 
# cv.VideoCapture --> arg: either integer that means from device camera, 
#                          or already present video
capture = cv.VideoCapture("funny.mp4")

while True:
  isTrue, frame = capture.read() # Read the video frame by frame 
  cv.imshow('Video', frame)

  if cv.waitKey(20) & 0xFF==ord('d'):
    break

# Release the capture pointer
capture.release()
cv.destroyAllWindows()