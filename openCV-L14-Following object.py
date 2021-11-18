import cv2
import numpy as np
print(cv2.__version__)

CAM_ID = 0
CAM_FPS = 30
CAM_RES = (640, 480)

SCREEN_RESOLUTION = (1980, 1080)

WINDOW_CAMERA_POS = (0, 0)
WINDOW_CAMERA_NAME = 'Camera'

WINDOW_TRACKER_NAME = 'Trackbar'
WINDOW_TRACKER_DIM = (650, 500)

WINDOW_MASK_1_NAME = 'Mask1'
WINDOW_MASK_2_NAME = 'Mask2'
WINDOW_MASK_MERGED_NAME = 'Mask'
WINDOW_OBJ_NAME = 'Object'

hue1Min = 15
hue1Max = 30
hue2Min = 25
hue2Max = 30
satMin = 160
satMax = 255
valMin = 30
valMax = 255
minArea = 500

displaySetup = False
isSetupVisible = False


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


def onMinAreaChanged(val):
    global minArea
    minArea = 1 if val < 1 else val


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

while True:
    if displaySetup and not isSetupVisible:
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
        cv2.createTrackbar('Area (min)', WINDOW_TRACKER_NAME, minArea, 2000, onMinAreaChanged)

        isSetupVisible = True

    if not displaySetup and isSetupVisible:
        cv2.destroyWindow(WINDOW_TRACKER_NAME)
        cv2.destroyWindow(WINDOW_MASK_1_NAME)
        cv2.destroyWindow(WINDOW_MASK_2_NAME)
        cv2.destroyWindow(WINDOW_MASK_MERGED_NAME)
        cv2.destroyWindow(WINDOW_OBJ_NAME)
        isSetupVisible = False

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

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours (index = -1)
    #cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

    # Filter contours keeping only a certain size
    for c in contours:
        area = cv2.contourArea(c)
        if area >= minArea:
            #cv2.drawContours(frame, [c], 0, (255,0, 0), 3)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

            # m00 = count of all non-zeros pixel (which is the area)
            # m10 = sum of all non-zero pixels (x-axis)
            # m01 = sum of all non-zero pixels (y-axis)
            m = cv2.moments(c)
            centroid = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
            cv2.circle(frame, centroid, 7, (255, 255, 255), -1)

            # Convert centroid distance as a % from origin and apply this
            # percentage to the center of the window
            distPercent = (centroid[0] / frameDim[0], centroid[1] / frameDim[1])
            windowPosX = 0 + int((SCREEN_RESOLUTION[0] * distPercent[0]) / 2)
            windowPosY = 0 + int((SCREEN_RESOLUTION[1] * distPercent[1]) / 2)
            cv2.moveWindow(WINDOW_CAMERA_NAME, windowPosX, windowPosY)

    cv2.imshow(WINDOW_CAMERA_NAME, frame)

    if isSetupVisible:
        cv2.imshow(WINDOW_MASK_1_NAME, mask1Resized)
        cv2.moveWindow(WINDOW_MASK_1_NAME, WINDOW_CAMERA_POS[0], WINDOW_CAMERA_POS[1] + WINDOW_TRACKER_DIM[1] + 30)

        cv2.imshow(WINDOW_MASK_2_NAME, mask2Resized)
        cv2.moveWindow(WINDOW_MASK_2_NAME, WINDOW_CAMERA_POS[0] + len(maskResized[0]), WINDOW_CAMERA_POS[1] + WINDOW_TRACKER_DIM[1] + 30)

        cv2.imshow(WINDOW_MASK_MERGED_NAME, maskResized)
        cv2.moveWindow(WINDOW_MASK_MERGED_NAME, WINDOW_CAMERA_POS[0] + frameDim[0], WINDOW_CAMERA_POS[1] + WINDOW_TRACKER_DIM[1] + 30)

        cv2.imshow(WINDOW_OBJ_NAME, objResized)
        cv2.moveWindow(WINDOW_OBJ_NAME, WINDOW_CAMERA_POS[0] + frameDim[0] + len(maskResized[0]), WINDOW_CAMERA_POS[1] + WINDOW_TRACKER_DIM[1] + 30)

    key = cv2.waitKey(1)

    if key & 0xff == ord('q'):
        break

    if key & 0xff == ord('s'):
        displaySetup = True if displaySetup == False else False

cam.release()
cv2.destroyAllWindows()
