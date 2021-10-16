import cv2
print(cv2.__version__)

screenScaleFactor = 1.25
screenWidth = 1920
screenHeight = 1080

requestedFrameWidth = 160
requestedFrameHeight = 120

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, requestedFrameWidth)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, requestedFrameHeight)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

effectiveFrameWidth = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
effectiveFrameHeight = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f'Requested frame dimensions was {requestedFrameWidth}x{requestedFrameHeight}.')
print(f'Effective frame dimensions is {effectiveFrameWidth}x{effectiveFrameHeight}.')

colNumber = int(screenWidth / screenScaleFactor / effectiveFrameWidth)
rowNumber = int(screenHeight / screenScaleFactor / effectiveFrameHeight)

windows = []
for col in range(0, colNumber):
    for row in range(0, rowNumber):
        windowName = f'cam ({str(row)},{str(col)})'
        windows.append(windowName)
        cv2.namedWindow(windowName, cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(windowName, col * effectiveFrameWidth, row * effectiveFrameHeight)
        cv2.resizeWindow(windowName, effectiveFrameWidth, effectiveFrameHeight)

while True:
    _, frame = cam.read()
    for w in windows:
        cv2.imshow(w, frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
