import cv2
import random
print(cv2.__version__)

CAM_ID = 0
CAM_FPS = 30

WINDOW_DIM = (640, 480)
WINDOW_POS = (0, 0)
WINDOW_NAME = 'Camera'

ROI_DIM = (100, 100)
ROI_MAX_VELOCITY = 5

# Setup camera
cam = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_DIM[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_DIM[1])
cam.set(cv2.CAP_PROP_FPS, CAM_FPS)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Display camera settings dialog
#cam.set(cv2.CAP_PROP_SETTINGS, 0)

# Calculate actual frame dimensions. These dimensions may be different than
# our settings as the camera supports only a subset of possible resolutions.
frameDim = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Setup the window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
cv2.moveWindow(WINDOW_NAME, WINDOW_POS[0], WINDOW_POS[1])
cv2.resizeWindow(WINDOW_NAME, frameDim[0], frameDim[1])

# Set ROI's initial position (roiX and roiY) and velocity (roiVX and roiVY)
roiXY = (
    random.randint(0, frameDim[0] - 1 - ROI_DIM[0]),
    random.randint(0, frameDim[1] - 1 - ROI_DIM[1]))
roiVXVY = (
    random.randint(ROI_MAX_VELOCITY * -1, ROI_MAX_VELOCITY),
    random.randint(ROI_MAX_VELOCITY * -1, ROI_MAX_VELOCITY)
)

while True:
    # Capture frame
    _, frame = cam.read()
    frameGray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # Move region of interest
    roiXY = (roiXY[0] + roiVXVY[0], roiXY[1] + roiVXVY[1])

    # Detect collision on X-axis
    if roiXY[0] <= 0:
        roiXY = (0, roiXY[1])
        roiVXVY = (roiVXVY[0] * -1, roiVXVY[1])
    elif roiXY[0] + ROI_DIM[0] >= frameDim[0] - 1:
        roiXY = (frameDim[0] - 1 - ROI_DIM[0], roiXY[1])
        roiVXVY = (roiVXVY[0] * -1, roiVXVY[1])

    # Detect collision on Y-axis
    if roiXY[1] <= 0:
        roiXY = (roiXY[0], 0)
        roiVXVY = (roiVXVY[0], roiVXVY[1] * -1)
    elif roiXY[1] + ROI_DIM[1] >= frameDim[1] - 1:
        roiXY = (roiXY[0], frameDim[1] - 1 - ROI_DIM[1])
        roiVXVY = (roiVXVY[0], roiVXVY[1] * -1)

    # Extract ROI
    frameROI = frame[
        roiXY[1]:roiXY[1] + ROI_DIM[1],
        roiXY[0]:roiXY[0] + ROI_DIM[0],
    ]

    # Add ROI to the frame
    frameGray[
        roiXY[1]:roiXY[1] + ROI_DIM[1],
        roiXY[0]:roiXY[0] + ROI_DIM[0],
    ] = frameROI

    cv2.imshow(WINDOW_NAME, frameGray)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
