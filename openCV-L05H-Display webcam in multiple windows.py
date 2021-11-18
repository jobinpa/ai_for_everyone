import cv2

print(cv2.__version__)

CAM_ID = 0

cam = cv2.VideoCapture(CAM_ID)

while True:
    _, frame = cam.read()

    scaleFactor = 0.8
    scaledWidth = int(frame.shape[1] * scaleFactor)
    scaledHeight = int(frame.shape[0] * scaleFactor)
    scaledFrame = (scaledWidth, scaledHeight)

    resized = cv2.resize(frame, scaledFrame, interpolation=cv2.INTER_AREA)
    grayFrame = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    windowWidth = scaledWidth
    windowHeight = scaledHeight

    cv2.namedWindow('Window1')
    cv2.moveWindow('Window1', 0, 0)
    cv2.imshow('Window1', resized)

    cv2.namedWindow('Window2')
    cv2.moveWindow('Window2', 0 + windowWidth, 0)
    cv2.imshow('Window2', grayFrame)

    cv2.namedWindow('Window3')
    cv2.moveWindow('Window3', 0, 0 + windowHeight)
    cv2.imshow('Window3', grayFrame)

    cv2.namedWindow('Window4')
    cv2.moveWindow('Window4', 0 + windowWidth, 0 + windowHeight)
    cv2.imshow('Window4', resized)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
