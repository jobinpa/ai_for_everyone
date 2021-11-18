import cv2
import time
print(cv2.__version__)

CAM_ID = 0
CAM_FPS = 30
CAM_RES = (640, 480)

WINDOW_CAMERA_POS = (0, 0)
WINDOW_CAMERA_NAME = 'Camera'


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


# Setup camera
cam = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RES[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RES[1])
cam.set(cv2.CAP_PROP_FPS, CAM_FPS)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Initialize fps
fps = int(cam.get(cv2.CAP_PROP_FPS))

# Calculate actual frame dimensions. These dimensions may be different than
# our settings as the camera supports only a subset of possible resolutions.
frameDim = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Setup the window
cv2.namedWindow(WINDOW_CAMERA_NAME)
cv2.moveWindow(WINDOW_CAMERA_NAME, WINDOW_CAMERA_POS[0], WINDOW_CAMERA_POS[1])
cv2.resizeWindow(WINDOW_CAMERA_NAME, frameDim[0], frameDim[1])

# Setup face and eye detection
faceCascade = cv2.CascadeClassifier('./haar/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('./haar/haarcascade_eye.xml')

previousTime = -1

while True:
    # Calculate fps
    now = time.perf_counter()
    if previousTime >= 0:
        dT = 0.0000001 if now == previousTime else now - previousTime
        unfilteredFps = 1 / dT
        # Apply low-pass filter to reduce noise
        fps = fps * 0.98 + unfilteredFps * 0.02
    previousTime = now

    # Read camera
    _, frame = cam.read()

    # Convert frame to gray scale  to impove model perf
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(frameGray, 1.3, 5)
    for faceX, faceY, faceWidth, faceHeight in faces:
        cv2.rectangle(frame, (faceX, faceY), (faceX+faceWidth, faceY+faceHeight), (255, 0, 0), 3)

        # We look for eyes inside faces to improve performance
        faceRoi = frame[faceY:faceY + faceHeight, faceX:faceX + faceWidth]
        faceRoiGray = cv2.cvtColor(faceRoi, cv2.COLOR_BGR2GRAY)
        eyes = eyeCascade.detectMultiScale(faceRoiGray, 1.3, 5)

        for relEyeX, relEyeY, eyeWidth, eyeHeight in eyes:
            eyeX = relEyeX + faceX
            eyeY = relEyeY + faceY
            cv2.rectangle(frame, (eyeX, eyeY), (eyeX+eyeWidth, eyeY+eyeHeight), (0, 255, 0), 3)

    # Add fps control
    displayFps(frame, fps)

    cv2.imshow(WINDOW_CAMERA_NAME, frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
