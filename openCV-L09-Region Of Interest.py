import cv2
import numpy as np
print(cv2.__version__)

CAM_ID = 0
CAM_FPS = 30

WINDOW_DIM = (640, 480)
WINDOW_POS = (0, 0)
WINDOW_NAME = 'Camera'

ROI_SIZE_PERCENT = 0.50
ROI_WINDOW_NAME = 'ROI'
ROI_GRAY_WINDOW_NAME = 'ROI (Gray)'

cam = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_DIM[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_DIM[1])
cam.set(cv2.CAP_PROP_FPS, CAM_FPS)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

centerX = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
centerY = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
cv2.moveWindow(WINDOW_NAME, WINDOW_POS[0], WINDOW_POS[1])
cv2.resizeWindow(WINDOW_NAME, int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

roiWidth = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) * ROI_SIZE_PERCENT)
roiHeight = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) * ROI_SIZE_PERCENT)

cv2.namedWindow(ROI_WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
cv2.moveWindow(ROI_WINDOW_NAME, WINDOW_POS[0] + int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), WINDOW_POS[1])
cv2.resizeWindow(ROI_WINDOW_NAME, roiWidth, roiHeight)

cv2.namedWindow(ROI_GRAY_WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
cv2.moveWindow(ROI_GRAY_WINDOW_NAME, WINDOW_POS[0] + int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), WINDOW_POS[1] + roiHeight)
cv2.resizeWindow(ROI_GRAY_WINDOW_NAME, roiWidth, roiHeight)

while True:
    _, frame = cam.read()

    halfRectHeight = int(roiWidth / 2)
    halfRectWidth = int(roiHeight / 2)

    # Creating an array from the slice so the data is independant from
    # the original pictire. If we don't, the content of the slice will
    # get updated as the frame is updated.
    frameROI = np.array(frame[
        centerY - halfRectHeight:centerY + halfRectHeight,
        centerX - halfRectWidth: centerX + halfRectWidth
    ])

    frameRoiGray = cv2.cvtColor(frameROI, cv2.COLOR_BGR2GRAY)

    # This would modify the frameROI slice if we hadn't created an
    # independant array using np.array
    frame[
        centerY - halfRectHeight:centerY + halfRectHeight,
        centerX - halfRectWidth: centerX + halfRectWidth
    ] = cv2.cvtColor(frameRoiGray, cv2.COLOR_GRAY2BGR)

    cv2.imshow(WINDOW_NAME, frame)
    cv2.imshow(ROI_WINDOW_NAME, frameROI)
    cv2.imshow(ROI_GRAY_WINDOW_NAME, frameRoiGray)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
