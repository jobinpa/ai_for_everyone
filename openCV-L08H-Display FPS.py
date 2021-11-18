import cv2
import time
print(cv2.__version__)

CAM_ID = 0
CAM_FPS = 30

WINDOW_DIM = (640, 480)
WINDOW_POS = (0, 0)
WINDOW_NAME = 'Camera'


def displayFps(frame, fps):
    FPS_TEXT_FONT = cv2.FONT_HERSHEY_COMPLEX
    FPS_TEXT_COLOR = (0, 0, 255)
    FPS_TEXT_THICKNESS = 1
    FPS_TEXT_SCALE = 0.60
    FPS_BACKGROUND_COLOR = (0, 0, 0)

    text = f'{round(fps)} fps'
    (textWidth, textHeight), baseline = cv2.getTextSize(text, FPS_TEXT_FONT, FPS_TEXT_SCALE, FPS_TEXT_THICKNESS)
    frameDim = (len(frame[0]), len(frame))
    textX = frameDim[0] - textWidth
    textY = 0 + textHeight
    cv2.rectangle(frame, (textX, 0), (frameDim[0], textHeight + baseline), FPS_BACKGROUND_COLOR, -1)
    cv2.putText(frame, text, (textX, textY), FPS_TEXT_FONT, FPS_TEXT_SCALE, FPS_TEXT_COLOR, FPS_TEXT_THICKNESS)


cam = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_DIM[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_DIM[1])
cam.set(cv2.CAP_PROP_FPS, CAM_FPS)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
cv2.moveWindow(WINDOW_NAME, WINDOW_POS[0], WINDOW_POS[1])
cv2.resizeWindow(WINDOW_NAME, int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fps = int(cam.get(cv2.CAP_PROP_FPS))
previousTime = -1

while True:
    # Calculate fps
    now = time.perf_counter()
    if previousTime >= 0:
        dT = 0.0000001 if now == previousTime else now - previousTime
        unfilteredFps = 1 / dT
        # Apply low-pass filter to reduce noise
        fps = fps * 0.98 + unfilteredFps * 0.02
        print(f'Raw FPS: {unfilteredFps} / Filtered FPS: {fps}')
    previousTime = now

    _, frame = cam.read()

    # Add fps control
    displayFps(frame, fps)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
