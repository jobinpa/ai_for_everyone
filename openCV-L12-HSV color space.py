import cv2
import numpy as np
print(cv2.__version__)

# HSV        = Hue, Saturation, Value
# Hue        = Color (0 to 179)
# Saturation = Whiteness/Color dilution (0 = white, 255 = no white)
# Value      = Blackiness/Luminosity (0 = black, 255 = no black)

CAM_ID = 0
CAM_FPS = 30
CAM_RES = (640, 480)

WINDOW_CAMERA_POS = (0, 0)
WINDOW_CAMERA_NAME = 'Camera'

WINDOW_COLOR_SAMPLE_POS = (0, 0)
WINDOW_COLOR_SAMPLE_NAME = 'Color'
WINDOW_COLOR_SAMPLE_DIM = (250, 250)

isColorSelecting = False
isColorSelected = False
clickPos = (-1, -1)
colorBGR = [0, 0, 0]
colorHSV = cv2.cvtColor(np.full([1, 1, 3], colorBGR, dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]


def mouseClick(event, xPos, yPos, flags, params):
    global isColorSelecting, clickPos
    clickPos = (xPos, yPos)

    if event == cv2.EVENT_LBUTTONDOWN:
        isColorSelecting = True


def displayText(frame, text, line):
    (textWidth, textHeight), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
    frameDim = (len(frame[0]), len(frame))
    textX = frameDim[0] - textWidth
    textY = (line * baseline) + (line + 1) * textHeight
    cv2.rectangle(frame, (textX, textY - textHeight), (frameDim[0], textY + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (textX, textY), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)


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

# Setup the windows
cv2.namedWindow(WINDOW_CAMERA_NAME)
cv2.moveWindow(WINDOW_CAMERA_NAME, WINDOW_CAMERA_POS[0], WINDOW_CAMERA_POS[1])
cv2.resizeWindow(WINDOW_CAMERA_NAME, frameDim[0], frameDim[1])

cv2.setMouseCallback(WINDOW_CAMERA_NAME, mouseClick)

while True:
    _, frame = cam.read()

    if isColorSelecting:
        colorBGR = frame[clickPos[1], clickPos[0]]
        colorHSV = cv2.cvtColor(np.full([1, 1, 3], colorBGR, dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
        isColorSelecting = False
        print(f'Just selected color. BGR = {colorBGR}, HSV = {colorHSV}.')

        if not isColorSelected:
            cv2.namedWindow(WINDOW_COLOR_SAMPLE_NAME)
            cv2.moveWindow(WINDOW_COLOR_SAMPLE_NAME, WINDOW_CAMERA_POS[0] + frameDim[0], WINDOW_CAMERA_POS[1])
            cv2.resizeWindow(WINDOW_COLOR_SAMPLE_NAME, WINDOW_COLOR_SAMPLE_DIM[0], WINDOW_COLOR_SAMPLE_DIM[1])
            isColorSelected = True

    if isColorSelected:
        colorFrame = np.full([WINDOW_COLOR_SAMPLE_DIM[0], WINDOW_COLOR_SAMPLE_DIM[1], 3], colorBGR, np.uint8)
        displayText(colorFrame, f'BGR: {colorBGR}', 0)
        displayText(colorFrame, f'HSV: {colorHSV}', 1)
        cv2.imshow(WINDOW_COLOR_SAMPLE_NAME, colorFrame)

    cv2.imshow(WINDOW_CAMERA_NAME, frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
