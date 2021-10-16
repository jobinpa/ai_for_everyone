import cv2
print(cv2.__version__)

CAM_ID = 1
CAM_FPS = 30

WINDOW_DIM = (640, 480)
WINDOW_POS = (0, 0)
WINDOW_NAME = 'Camera'

evt = -1
pnt = (0, 0)


def mouseClick(event, xPos, yPos, flags, params):
    global evt
    global pnt

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.cv2.EVENT_LBUTTONUP:
        evt = event
        pnt = (xPos, yPos)
    elif event == cv2.EVENT_RBUTTONUP:
        evt = event
        pnt = (xPos, yPos)

    print(f'Mouse event was {event} at position ({xPos},{yPos})')


# Setup camera
cam = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_DIM[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_DIM[1])
cam.set(cv2.CAP_PROP_FPS, CAM_FPS)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Calculate actual frame dimensions. These dimensions may be different than
# our settings as the camera supports only a subset of possible resolutions.
frameDim = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Setup the window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
cv2.moveWindow(WINDOW_NAME, WINDOW_POS[0], WINDOW_POS[1])
cv2.resizeWindow(WINDOW_NAME, frameDim[0], frameDim[1])

cv2.setMouseCallback(WINDOW_NAME, mouseClick)

while True:
    _, frame = cam.read()

    if evt == cv2.EVENT_LBUTTONDOWN or evt == cv2.EVENT_LBUTTONUP:
        cv2.circle(frame, pnt, 25, (255, 0, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
