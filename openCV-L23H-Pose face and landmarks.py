import cv2
import mediapipehelper as mph
import numpy as np
print(f'OpenCV version is {cv2.__version__}')

# Parameters
CAM_ID = 0
CAM_FPS = 30
CAM_RES = (640, 480)

FLIP_CAMERA_FRAME_HORIZONTALY = True

WINDOW_CAMERA_POS = (0, 0)
WINDOW_CAMERA_NAME = 'Camera'
WINDOW_BLACK_NAME = 'Black'

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

cv2.namedWindow(WINDOW_BLACK_NAME)
cv2.moveWindow(WINDOW_BLACK_NAME, WINDOW_CAMERA_POS[0] + frameDim[0], WINDOW_CAMERA_POS[1])
cv2.resizeWindow(WINDOW_BLACK_NAME, frameDim[0], frameDim[1])

print('Press "q" to quit...')

faceDetection = mph.FaceMeshDetection(min_detection_confidence=0.5, max_num_faces=5)
faceRegions = [
    mph.FACEMESH_REGION_SILHOUETTE,
    mph.FACEMESH_REGION_LEFT_EYE_UPPER_0,
    mph.FACEMESH_REGION_LEFT_EYE_LOWER_0,
    mph.FACEMESH_REGION_RIGHT_EYE_UPPER_0,
    mph.FACEMESH_REGION_RIGHT_EYE_LOWER_0,
    mph.FACEMESH_REGION_LIPS_LOWER_OUTER,
    mph.FACEMESH_REGION_LIPS_LOWER_INNER,
    mph.FACEMESH_REGION_LIPS_UPPER_OUTER,
    mph.FACEMESH_REGION_LIPS_UPPER_INNER,
    mph.FACEMESH_REGION_NOSE_BOTTOM,
    mph.FACEMESH_REGION_NOSE_LEFT_CORNER,
    mph.FACEMESH_REGION_NOSE_RIGHT_CORNER,
    mph.FACEMESH_REGION_NOSE_TIP,
    mph.FACEMESH_REGION_MIDWAY_BETWEEN_EYES
]

# Read and display camera capture
while True:
    _, frame = cam.read()

    if FLIP_CAMERA_FRAME_HORIZONTALY:
        frame = cv2.flip(frame, 1)
    frameBlack = np.zeros([frameDim[1], frameDim[0], 3], dtype=np.uint8)

    for faceMesh in faceDetection.detectFaces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)):
        for lm in faceMesh.getLandmarks(*faceRegions):
            cv2.circle(frame, lm, 1, (0, 0, 255), -1)
            cv2.circle(frameBlack, lm, 1, (0, 0, 255), -1)

    cv2.imshow(WINDOW_CAMERA_NAME, frame)
    cv2.imshow(WINDOW_BLACK_NAME, frameBlack)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Cleanup
print('Shutting down...')
cam.release()
cv2.destroyAllWindows()
print('Application ended')
