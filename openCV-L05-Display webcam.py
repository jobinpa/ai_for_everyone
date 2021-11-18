import cv2

print(cv2.__version__)

CAM_ID = 0

cam = cv2.VideoCapture(CAM_ID)

while True:
    _, frame = cam.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('my WEBcam', grayFrame)
    cv2.moveWindow('my WEBcam', 0, 0)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
