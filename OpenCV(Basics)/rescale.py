import cv2 as cv

def rescaleFrame(frame, scale=0.75):
    # Works for image, videos and live videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Reading Video
capture = cv.VideoCapture('funny.mp4')

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame)

    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

# Release the capture pointer
capture.release()
cv.destroyAllWindows()


# Similary for img
# Read the image 
img = cv.imread('000007.jpg')

# Resize the img
resized_img = rescaleFrame(img, scale=0.2)

# Display the read image and resized image
cv.imshow('Car', img)
cv.imshow('Resized Car', resized_img)

cv.waitKey(0)

