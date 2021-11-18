import cv2
print(cv2.__version__)

CAM_ID = 0
CAM_FPS = 30

WINDOW_MAIN_DIM = (640, 480)
WINDOW_MAIN_POS = (0, 0)
WINDOW_MAIN_NAME = 'Camera'

WINDOW_SELECTION_NAME = 'Selection'

frameDim = (0, 0)

selectionWindowVisible = False
selectionInProgress = False
selectionCompleted = False
selectedAreaPnt1 = (-1, -1)
selectedAreaPnt2 = (-1, -1)


def mouseClick(event, xPos, yPos, flags, params):
    global frameDim, selectionWindowVisible, selectionInProgress
    global selectionCompleted, selectedAreaPnt1, selectedAreaPnt2

    canStartSelection = not selectionInProgress and not selectionCompleted

    if event == cv2.EVENT_LBUTTONDOWN and canStartSelection:
        selectionInProgress = True
        selectedAreaPnt1 = (xPos, yPos)
        selectedAreaPnt2 = (xPos, yPos)

    if event == cv2.EVENT_MOUSEMOVE and selectionInProgress:
        selectedAreaPnt2 = (
            xPos if xPos < frameDim[0] else frameDim[0] - 1,
            yPos if yPos < frameDim[1] else frameDim[1] - 1
        )

    if event == cv2.EVENT_LBUTTONUP and selectionInProgress:
        selectionInProgress = False
        selectionCompleted = True

    if event == cv2.EVENT_RBUTTONDOWN and selectionCompleted:
        selectedAreaPnt1 = (-1, -1)
        selectedAreaPnt2 = (-1, -1)
        selectionCompleted = False
        if selectionWindowVisible:
            cv2.destroyWindow(WINDOW_SELECTION_NAME)
            selectionWindowVisible = False


# Setup camera
cam = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_MAIN_DIM[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_MAIN_DIM[1])
cam.set(cv2.CAP_PROP_FPS, CAM_FPS)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Calculate actual frame dimensions. These dimensions may be different than
# our settings as the camera supports only a subset of possible resolutions.
frameDim = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Setup the window
cv2.namedWindow(WINDOW_MAIN_NAME, cv2.WINDOW_GUI_NORMAL)
cv2.moveWindow(WINDOW_MAIN_NAME, WINDOW_MAIN_POS[0], WINDOW_MAIN_POS[1])
cv2.resizeWindow(WINDOW_MAIN_NAME, frameDim[0], frameDim[1])
cv2.setMouseCallback(WINDOW_MAIN_NAME, mouseClick)

while True:
    _, frame = cam.read()

    if selectionCompleted:
        startRow = selectedAreaPnt1[1] if selectedAreaPnt1[1] < selectedAreaPnt2[1] else selectedAreaPnt2[1]
        endRow = selectedAreaPnt2[1] if selectedAreaPnt2[1] > selectedAreaPnt1[1] else selectedAreaPnt1[1]
        startCol = selectedAreaPnt1[0] if selectedAreaPnt1[0] < selectedAreaPnt2[0] else selectedAreaPnt2[0]
        endCol = selectedAreaPnt2[0] if selectedAreaPnt2[0] > selectedAreaPnt1[0] else selectedAreaPnt1[0]
        cv2.imshow(WINDOW_SELECTION_NAME, frame[startRow:endRow + 1, startCol:endCol + 1])
        cv2.moveWindow(WINDOW_SELECTION_NAME, WINDOW_MAIN_POS[0] + frameDim[0], WINDOW_MAIN_POS[1])
        selectionWindowVisible = True

    if selectionInProgress or selectionCompleted:
        cv2.rectangle(frame, selectedAreaPnt1, selectedAreaPnt2, (255, 0, 0), 2)

    cv2.imshow(WINDOW_MAIN_NAME, frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
