import cv2 as cv

def changeRes(width, height):
    # Change resolution of LIVE VIDEO
    capture.set(2, width)
    capture.set(4, height)

# Reading Video
capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()

    frame_resized = changeRes(frame.shape[1], frame.shape[0])

    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

# Release the capture pointer
capture.release()
cv.destroyAllWindows()