import cv2
print(cv2.__version__)

CAM_ID = 1
CAM_FPS = 30
CAM_RES = (640, 480)

SCREEN_RESOLUTION = (1980, 1080)
SCREEN_SCALE_FACTOR = (1.25)

WINDOW_CAMERA_POS = (0, 0)
WINDOW_CAMERA_NAME = 'Camera'

WINDOW_SETTINGS_DIM = (400, 150)
WINDOW_SETTINGS_NAME = 'Settings'

TRACKBAR_WIDTH_NAME = 'Width'
TRACKBAR_HEIGHT_NAME = 'Height'
TRACKBAR_XPOS_NAME = 'Pos-X'
TRACKBAR_YPOS_NAME = 'Pos-Y'

sacledFrameDim = (CAM_RES[0], CAM_RES[1])
frameRatio = CAM_RES[0] / CAM_RES[1]
eventHandlerDisabled = False


def onWidthChanged(width):
    global sacledFrameDim, frameRatio, eventHandlerDisabled

    if eventHandlerDisabled:
        return

    scaledWidth = width
    scaledHeight = round(scaledWidth / frameRatio)

    if scaledWidth > 0 and scaledHeight > 0:
        sacledFrameDim = (scaledWidth, scaledHeight)


def onHeightChanged(height):
    global sacledFrameDim, frameRatio, eventHandlerDisabled

    if eventHandlerDisabled:
        return

    scaledWidth = round(height * frameRatio)
    scaledHeight = height

    if scaledWidth > 0 and scaledHeight > 0:
        sacledFrameDim = (scaledWidth, scaledHeight)


def onPosXChanged(posX):
    y = cv2.getTrackbarPos(TRACKBAR_YPOS_NAME, WINDOW_SETTINGS_NAME)
    cv2.moveWindow(WINDOW_CAMERA_NAME, posX, y)


def onPosYChanged(posY):
    x = cv2.getTrackbarPos(TRACKBAR_XPOS_NAME, WINDOW_SETTINGS_NAME)
    cv2.moveWindow(WINDOW_CAMERA_NAME, x, posY)


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
sacledFrameDim = (frameDim[0], frameDim[1])
frameRatio = frameDim[0] / frameDim[1]

# Setup the camera window
cv2.namedWindow(WINDOW_CAMERA_NAME)
cv2.moveWindow(WINDOW_CAMERA_NAME, WINDOW_CAMERA_POS[0], WINDOW_CAMERA_POS[1])
cv2.resizeWindow(WINDOW_CAMERA_NAME, frameDim[0], frameDim[1])

# Setup the settings window
cv2.namedWindow(WINDOW_SETTINGS_NAME)
cv2.moveWindow(WINDOW_SETTINGS_NAME, WINDOW_CAMERA_POS[0] + frameDim[0], WINDOW_CAMERA_POS[1])
cv2.resizeWindow(WINDOW_SETTINGS_NAME, WINDOW_SETTINGS_DIM[0], WINDOW_SETTINGS_DIM[1])

# Add settings control
cv2.createTrackbar(TRACKBAR_WIDTH_NAME, WINDOW_SETTINGS_NAME, frameDim[0], frameDim[0], onWidthChanged)
cv2.createTrackbar(TRACKBAR_HEIGHT_NAME, WINDOW_SETTINGS_NAME, frameDim[1], frameDim[1], onHeightChanged)
cv2.createTrackbar(TRACKBAR_XPOS_NAME, WINDOW_SETTINGS_NAME, WINDOW_CAMERA_POS[0], int(SCREEN_RESOLUTION[0]/SCREEN_SCALE_FACTOR), onPosXChanged)
cv2.createTrackbar(TRACKBAR_YPOS_NAME, WINDOW_SETTINGS_NAME, WINDOW_CAMERA_POS[1], int(SCREEN_RESOLUTION[1]/SCREEN_SCALE_FACTOR), onPosYChanged)

while True:
    _, frame = cam.read()

    cv2.resizeWindow(WINDOW_CAMERA_NAME, sacledFrameDim[0], sacledFrameDim[1])

    # Window have minimum size. If the size of the window after the resize
    # operation is different to the expected size, we have reach this limit
    # and we need to recalculate the scaled size based on this limit.
    _, _, wWidth, wHeight = cv2.getWindowImageRect(WINDOW_CAMERA_NAME)
    if wWidth != sacledFrameDim[0] or wHeight != sacledFrameDim[1]:
        sacledFrameDim = (wWidth, round(wWidth / frameRatio))
        cv2.resizeWindow(WINDOW_CAMERA_NAME, sacledFrameDim[0], sacledFrameDim[1])

    scaledFrame = cv2.resize(frame, sacledFrameDim)
    cv2.imshow(WINDOW_CAMERA_NAME, scaledFrame)

    # As setTrackbarPos triggers the event handler, we want to temporary
    # disable them.
    eventHandlerDisabled = True
    cv2.setTrackbarPos(TRACKBAR_WIDTH_NAME, WINDOW_SETTINGS_NAME, sacledFrameDim[0])
    cv2.setTrackbarPos(TRACKBAR_HEIGHT_NAME, WINDOW_SETTINGS_NAME, sacledFrameDim[1])
    eventHandlerDisabled = False

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
