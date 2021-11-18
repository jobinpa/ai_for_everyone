import cv2
import mediapipe as mp

print(f'OpenCV version is {cv2.__version__}')

# Parameters
CAM_ID = 0
CAM_FPS = 30
CAM_RES = (640, 480)

FLIP_CAMERA_FRAME_HORIZONTALY = True

WINDOW_CAMERA_POS = (0, 0)
WINDOW_CAMERA_NAME = 'Camera'

FINGER_COLORS = [
    (0, 0, 0), (112, 112, 112), (25, 125, 254),
    (0, 255, 255), (121, 11, 210), (131, 16, 100)
]

FINGER_MARKER_RADIUS = 10


class MPHelper:
    def convertMultiHandLandmarksToCoordinates(multiHandLandmarks, frameDim):
        hands = []
        if multiHandLandmarks != None:
            for handLandmarks in multiHandLandmarks:
                coordinates = []
                for landmark in handLandmarks.landmark:
                    lx = int(landmark.x * frameDim[0])
                    ly = int(landmark.y * frameDim[1])
                    coordinates.append((lx, ly))
                hands.append(coordinates)
        return hands

    def getWrist(coordinates):
        return [coordinates[0]]

    def getThumb(coordinates):
        return coordinates[1:5]

    def getIndexFinger(coordinates):
        return coordinates[5:9]

    def getMiddleFinger(coordinates):
        return coordinates[9:13]

    def getRingFinger(coordinates):
        return coordinates[13:17]

    def getPinky(coordinates):
        return coordinates[17:21]


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

# Setup media pipe
# https://google.github.io/mediapipe/solutions/hands.html
handDetection = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

print('Press "q" to quit...')

# Read and display camera capture
while True:
    _, frame = cam.read()

    # MediaPipe assumes the input image is mirrored, i.e., taken with a
    # front-facing/selfie camera with images flipped horizontally. Some cameras
    # do this automatically, but most don't. We flip the frame so the right
    # hand returned by MediaPipe is really the right hand. See the following
    # link for more details.
    # https://google.github.io/mediapipe/solutions/hands.html#output
    if FLIP_CAMERA_FRAME_HORIZONTALY:
        frame = cv2.flip(frame, 1)

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Capture hands and add landmarks to frame
    hands = MPHelper.convertMultiHandLandmarksToCoordinates(
        handDetection.process(frameRGB).multi_hand_landmarks,
        frameDim)

    for h in hands:
        for p in MPHelper.getWrist(h):
            cv2.circle(frame, center=p, radius=FINGER_MARKER_RADIUS, color=FINGER_COLORS[0], thickness=cv2.FILLED)
        for p in MPHelper.getThumb(h):
            cv2.circle(frame, center=p, radius=FINGER_MARKER_RADIUS, color=FINGER_COLORS[1], thickness=cv2.FILLED)
        for p in MPHelper.getIndexFinger(h):
            cv2.circle(frame, center=p, radius=FINGER_MARKER_RADIUS, color=FINGER_COLORS[2], thickness=cv2.FILLED)
        for p in MPHelper.getMiddleFinger(h):
            cv2.circle(frame, center=p, radius=FINGER_MARKER_RADIUS, color=FINGER_COLORS[3], thickness=cv2.FILLED)
        for p in MPHelper.getRingFinger(h):
            cv2.circle(frame, center=p, radius=FINGER_MARKER_RADIUS, color=FINGER_COLORS[4], thickness=cv2.FILLED)
        for p in MPHelper.getPinky(h):
            cv2.circle(frame, center=p, radius=FINGER_MARKER_RADIUS, color=FINGER_COLORS[5], thickness=cv2.FILLED)

    cv2.imshow(WINDOW_CAMERA_NAME, frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Cleanup
print('Shutting down...')
cam.release()
cv2.destroyAllWindows()
print('Application ended')
