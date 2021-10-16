import cv2
print(cv2.__version__)

CAM_ID = 1
CAM_FPS = 30

WINDOW_MAIN_DIM = (640, 480)
WINDOW_MAIN_POS = (0, 0)
WINDOW_NAME = 'Camera'

TRACKBAR_XPOS_NAME = 'xPos'
TRACKBAR_YPOS_NAME = 'yPos'
TRACKBAR_RADIUS_NAME = 'radius'
TRACKBAR_THICKNESS_NAME = 'thickness'

WINDOW_SETTINGS_NAME = 'Settings'
WINDOW_SETTINGS_DIM = (400, 250)

circlePos = (0, 0)
circleRadius = 25
circleThickness = 1


def onXPosChange(xPos):
    global circlePos
    circlePos = (xPos, circlePos[1])


def onYPosChange(yPos):
    global circlePos
    circlePos = (circlePos[0], yPos)


def onRadiusChange(radius):
    global circleRadius
    circleRadius = radius


def onThicknessChange(thickness):
    global circleThickness
    if thickness == 0:
        thickness = -1
    circleThickness = thickness


# Setup camera
cam = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_MAIN_DIM[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_MAIN_DIM[1])
cam.set(cv2.CAP_PROP_FPS, CAM_FPS)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Calculate actual frame dimensions. These dimensions may be different than
# our settings as the camera supports only a subset of possible resolutions.
frameDim = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
circlePos = (round(frameDim[0] / 2), round(frameDim[1] / 2))

# Setup the main window
cv2.namedWindow(WINDOW_NAME)
cv2.moveWindow(WINDOW_NAME, WINDOW_MAIN_POS[0], WINDOW_MAIN_POS[1])
cv2.resizeWindow(WINDOW_NAME, frameDim[0], frameDim[1])

# Setup the trackbar window
cv2.namedWindow(WINDOW_SETTINGS_NAME)
cv2.moveWindow(WINDOW_SETTINGS_NAME, WINDOW_MAIN_POS[0] + frameDim[0], WINDOW_MAIN_POS[1])
cv2.resizeWindow(WINDOW_SETTINGS_NAME, WINDOW_SETTINGS_DIM[0], WINDOW_SETTINGS_DIM[1])

# Create the trackbar and put it in the window
cv2.createTrackbar(TRACKBAR_XPOS_NAME, WINDOW_SETTINGS_NAME, circlePos[0], frameDim[0] - 1, onXPosChange)
cv2.createTrackbar(TRACKBAR_YPOS_NAME, WINDOW_SETTINGS_NAME, circlePos[1], frameDim[1] - 1, onYPosChange)
cv2.createTrackbar(TRACKBAR_RADIUS_NAME, WINDOW_SETTINGS_NAME, circleRadius, round(frameDim[0] if frameDim[0] < frameDim[1] else frameDim[1] / 2), onRadiusChange)
cv2.createTrackbar(TRACKBAR_THICKNESS_NAME, WINDOW_SETTINGS_NAME, 0, 10, onThicknessChange)

while True:
    _, frame = cam.read()
    cv2.circle(frame, circlePos, circleRadius, (255, 0, 0), circleThickness)
    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
