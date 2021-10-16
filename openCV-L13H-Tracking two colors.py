import cv2
import numpy as np
print(cv2.__version__)

CAM_ID = 1
CAM_FPS = 30
CAM_RES = (640, 480)

WINDOW_CAMERA_POS = (0, 0)
WINDOW_CAMERA_NAME = 'Camera'

WINDOW_TRACKER_NAME = 'Trackbar'
WINDOW_TRACKER_DIM = (650, 500)

hue1Min = 10
hue1Max = 20
hue2Min = 10
hue2Max = 20
satMin = 10
satMax = 250
valMin = 10
valMax = 250


def onHue1MinChanged(hue):
    global hue1Min
    hue1Min = hue


def onHue1MaxChanged(hue):
    global hue1Max
    hue1Max = hue


def onHue2MinChanged(hue):
    global hue2Min
    hue2Min = hue


def onHue2MaxChanged(hue):
    global hue2Max
    hue2Max = hue


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

cv2.createTrackbar('Hue1 (min)', WINDOW_TRACKER_NAME, hue1Min, 179, onHue1MinChanged)
cv2.createTrackbar('Hue1 (max)', WINDOW_TRACKER_NAME, hue1Max, 179, onHue1MaxChanged)
cv2.createTrackbar('Hue2 (min)', WINDOW_TRACKER_NAME, hue2Min, 179, onHue2MinChanged)
cv2.createTrackbar('Hue2 (max)', WINDOW_TRACKER_NAME, hue2Max, 179, onHue2MaxChanged)
cv2.createTrackbar('Sat (min)', WINDOW_TRACKER_NAME, satMin, 255, onSatMinChanged)
cv2.createTrackbar('Sat (max)', WINDOW_TRACKER_NAME, satMax, 255, onSatMaxChanged)
cv2.createTrackbar('Val (min)', WINDOW_TRACKER_NAME, valMin, 255, onValMinChanged)
cv2.createTrackbar('Val (max)', WINDOW_TRACKER_NAME, valMax, 255, onValMaxChanged)

while True:
    _, frame = cam.read()

    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1Bound = np.array([hue1Min, satMin, valMin])
    upper1Bound = np.array([hue1Max, satMax, valMax])
    mask1 = cv2.inRange(frameHSV, lower1Bound, upper1Bound)
    mask1Resized = cv2.resize(mask1, (int(frameDim[0] / 2), int(frameDim[1] / 2)))

    lower2Bound = np.array([hue2Min, satMin, valMin])
    upper2Bound = np.array([hue2Max, satMax, valMax])
    mask2 = cv2.inRange(frameHSV, lower2Bound, upper2Bound)
    mask2Resized = cv2.resize(mask2, (int(frameDim[0] / 2), int(frameDim[1] / 2)))

    mask = mask1 | mask2
    maskResized = cv2.resize(mask, (int(frameDim[0] / 2), int(frameDim[1] / 2)))

    obj = cv2.bitwise_and(frame, frame, mask=mask)
    objResized = cv2.resize(obj, (int(frameDim[0] / 2), int(frameDim[1] / 2)))

    cv2.imshow(WINDOW_CAMERA_NAME, frame)

    cv2.imshow('Mask1', mask1Resized)
    cv2.moveWindow('Mask1', WINDOW_CAMERA_POS[0], WINDOW_CAMERA_POS[1] + WINDOW_TRACKER_DIM[1] + 30)

    cv2.imshow('Mask2', mask2Resized)
    cv2.moveWindow('Mask2', WINDOW_CAMERA_POS[0] + len(maskResized[0]), WINDOW_CAMERA_POS[1] + WINDOW_TRACKER_DIM[1] + 30)

    cv2.imshow('Mask', maskResized)
    cv2.moveWindow('Mask', WINDOW_CAMERA_POS[0] + frameDim[0], WINDOW_CAMERA_POS[1] + WINDOW_TRACKER_DIM[1] + 30)

    cv2.imshow('Object', objResized)
    cv2.moveWindow('Object', WINDOW_CAMERA_POS[0] + frameDim[0] + len(maskResized[0]), WINDOW_CAMERA_POS[1] + WINDOW_TRACKER_DIM[1] + 30)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
