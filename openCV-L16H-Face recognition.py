import cv2
import os
import time
import dlib
import face_recognition as fr
import numpy as np
from pathlib import Path

print(f'OpenCV version is {cv2.__version__}')

# Parameters
CAM_ID = 1
CAM_FPS = 30
CAM_RES = (640, 480)

FACE_RECOGNITION_TOLERANCE = 0.6  # Max distance from encoded face

WINDOW_CAMERA_POS = (0, 0)
WINDOW_CAMERA_NAME = 'Camera'

KNOWN_FACES_DIR = 'demoImages/known'
ENCODING_JITTERS = 1      # How many time to resample img during encoding.

FPS_SAMPLING_INTERVAL_SECONDS = 1

# hog (faster), cnn (more accurate)
FACE_DETECTION_MODEL = 'cnn' if dlib.DLIB_USE_CUDA else 'hog'
print(f'Face detection model is "{FACE_DETECTION_MODEL}".')

# Functions


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


def addFace(frame, name, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom + 30), (0, 0, 255), 2)
    cv2.rectangle(frame, (left, bottom + 30), (right, bottom), (0, 0, 255), -1)
    cv2.putText(frame, name, (left + 6, bottom + 24), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)


# Loading known faces
knownFaces = []
knownFaceNames = []

if not os.path.isdir(KNOWN_FACES_DIR):
    raise ValueError(f'Directory "{KNOWN_FACES_DIR}"" was not found.')

for filePath in Path(KNOWN_FACES_DIR).resolve().glob('*.jpg'):
    print(f'Loading face "{filePath}"')

    imgRgb = fr.load_image_file(filePath)
    imgFaceEncodings = fr.face_encodings(imgRgb, num_jitters=ENCODING_JITTERS)
    if len(imgFaceEncodings) != 1:
        raise ValueError(f'Image "{filePath}" contains {len(imgFaceEncodings)} face(s).')

    knownFaces.append(imgFaceEncodings[0])

    fileName = os.path.basename(filePath)
    knownFaceNames.append(fileName[:fileName.rfind('.')])

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

# Read and display camera capture
fps = int(cam.get(cv2.CAP_PROP_FPS))
frameCount = 0
fpsTimestamp = 0.0

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

    # Detect faces
    frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faceLocs = fr.face_locations(frameRgb, model=FACE_DETECTION_MODEL)
    faceEncodings = fr.face_encodings(frameRgb, faceLocs, num_jitters=ENCODING_JITTERS)

    # Run face recognition
    for loc, encoding in zip(faceLocs, faceEncodings):
        top, right, bottom, left = loc

        faceName = '???'
        faceDists = fr.face_distance(knownFaces, encoding)
        minDistIdx = np.argmin(faceDists)

        if faceDists[minDistIdx] <= FACE_RECOGNITION_TOLERANCE:
            faceName = knownFaceNames[minDistIdx]

        addFace(frame, faceName, left, top, right, bottom)

    # Add fps to frame
    addFps(frame, fps)

    # Display frame
    cv2.imshow(WINDOW_CAMERA_NAME, frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release camera and destroy all windows
cam.release()
cv2.destroyAllWindows()
