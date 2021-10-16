import cv2
import numpy as np
print(cv2.__version__)

CAM_ID = 1
CAM_FPS = 30
CAM_RES = (640, 480)

WINDOW_CAMERA_POS = (0, 0)
WINDOW_CAMERA_NAME = 'Camera'

WINDOW_TRACKER_NAME = 'Trackbar'
WINDOW_TRACKER_DIM = (500, 350)

hueMin = 10
hueMax = 20
satMin = 10
satMax = 250
valMin = 10
valMax = 250


def onHueMinChanged(hue):
    global hueMin
    hueMin = hue


def onHueMaxChanged(hue):
    global hueMax
    hueMax = hue


def onSatMinChanged(sat):
    global satMin
    satMin = sat


def onSatMaxChanged(sat):
    global satMax
    satMax = sat


def onValMinChanged(val):
    global valMin
    valMin = val


def onValMaxChanged(val):
    global valMax
    valMax = val


# Setup camera
cam = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RES[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RES[1])
cam.set(cv2.CAP_PROP_FPS, CAM_FPS)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Display camera settings dialog
#cam.set(cv2.CAP_PROP_SETTINGS, 0)

# Calculate actual frame dimensions. These dimensions may be different than
# our settings as the camera supports only a subset of possible resolutions.
frameDim = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Setup the window
cv2.namedWindow(WINDOW_CAMERA_NAME)
cv2.moveWindow(WINDOW_CAMERA_NAME, WINDOW_CAMERA_POS[0], WINDOW_CAMERA_POS[1])
cv2.resizeWindow(WINDOW_CAMERA_NAME, frameDim[0], frameDim[1])

cv2.namedWindow(WINDOW_TRACKER_NAME)
cv2.moveWindow(WINDOW_TRACKER_NAME, WINDOW_CAMERA_POS[0] + frameDim[0], WINDOW_CAMERA_POS[1])
cv2.resizeWindow(WINDOW_TRACKER_NAME, WINDOW_TRACKER_DIM[0], WINDOW_TRACKER_DIM[1])

cv2.createTrackbar('Hue (min)', WINDOW_TRACKER_NAME, hueMin, 179, onHueMinChanged)
cv2.createTrackbar('Hue (max)', WINDOW_TRACKER_NAME, hueMax, 179, onHueMaxChanged)
cv2.createTrackbar('Sat (min)', WINDOW_TRACKER_NAME, satMin, 255, onSatMinChanged)
cv2.createTrackbar('Sat (max)', WINDOW_TRACKER_NAME, satMax, 255, onSatMaxChanged)
cv2.createTrackbar('Val (min)', WINDOW_TRACKER_NAME, valMin, 255, onValMinChanged)
cv2.createTrackbar('Val (max)', WINDOW_TRACKER_NAME, valMax, 255, onValMaxChanged)

while True:
    _, frame = cam.read()

    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerBound = np.array([hueMin, satMin, valMin])
    upperBound = np.array([hueMax, satMax, valMax])

    mask = cv2.inRange(frameHSV, lowerBound, upperBound)
    maskResized = cv2.resize(mask, (int(frameDim[0] / 2), int(frameDim[1] / 2)))

    obj = cv2.bitwise_and(frame, frame, mask=mask)
    objResized = cv2.resize(obj, (int(frameDim[0] / 2), int(frameDim[1] / 2)))

    cv2.imshow(WINDOW_CAMERA_NAME, frame)

    cv2.imshow('Mask', maskResized)
    cv2.moveWindow('Mask', WINDOW_CAMERA_POS[0] + frameDim[0], WINDOW_CAMERA_POS[1] + WINDOW_TRACKER_DIM[1] + 30)

    cv2.imshow('Object', objResized)
    cv2.moveWindow('Object', WINDOW_CAMERA_POS[0] + frameDim[0] + len(maskResized[0]), WINDOW_CAMERA_POS[1] + WINDOW_TRACKER_DIM[1] + 30)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
