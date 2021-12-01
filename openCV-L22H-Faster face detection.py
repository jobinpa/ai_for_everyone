import cv2
import mediapipehelper as mph
import time

print(f'OpenCV version is {cv2.__version__}')

# Parameters
CAM_ID = 0
CAM_FPS = 30
CAM_RES = (640, 480)

FLIP_CAMERA_FRAME_HORIZONTALY = True

WINDOW_CAMERA_POS = (0, 0)
WINDOW_CAMERA_NAME = 'Camera'

FPS_SAMPLING_INTERVAL_SECONDS = 1


def addFps(frame, fps):
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
cam.set(cv2.CAP_PROP_ZOOM, 100)

# Display camera settings dialog
#cam.set(cv2.CAP_PROP_SETTINGS, 1)

# Calculate actual frame dimensions. These dimensions may be different than
# our settings as the camera supports only a subset of possible resolutions.
frameDim = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(f'Actual camera resolution is "{frameDim[0]}x{frameDim[1]}."')

# Setup the window
cv2.namedWindow(WINDOW_CAMERA_NAME)
cv2.moveWindow(WINDOW_CAMERA_NAME, WINDOW_CAMERA_POS[0], WINDOW_CAMERA_POS[1])
cv2.resizeWindow(WINDOW_CAMERA_NAME, frameDim[0], frameDim[1])

print('Press "q" to quit...')

faceDetection = mph.FaceDetection(min_detection_confidence=0.5)

# Initialize fps
fps = int(cam.get(cv2.CAP_PROP_FPS))
frameCount = 0
fpsTimestamp = 0.0

# Read and display camera capture
while True:
    # Calculate fps
    now = time.perf_counter()
    if (now - fpsTimestamp) >= FPS_SAMPLING_INTERVAL_SECONDS:
        fps = round(frameCount / (now - fpsTimestamp))
        fpsTimestamp = now
        frameCount = 0
    else:
        frameCount = frameCount + 1

    _, frame = cam.read()

    if FLIP_CAMERA_FRAME_HORIZONTALY:
        frame = cv2.flip(frame, 1)

    faces = faceDetection.detectFaces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for face in faces:
        cv2.rectangle(
            frame,
            face.getBoxUpperLeft(),
            face.getBoxLowerRight(),
            (0, 0, 255),
            5
        )

    # Add fps to frame
    addFps(frame, fps)

    cv2.imshow(WINDOW_CAMERA_NAME, frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Cleanup
print('Shutting down...')
cam.release()
cv2.destroyAllWindows()
print('Application ended')
