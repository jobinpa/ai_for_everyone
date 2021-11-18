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

HAND_MIN_DETECTION_CONFIDENCE = 0.7
HAND_MIN_TRACKING_CONFIDENCE = 0.5

FINGER_COLORS = [
    (0, 0, 0), (112, 112, 112), (25, 125, 254),
    (0, 255, 255), (121, 11, 210), (131, 16, 100)
]

FINGER_MARKER_RADIUS = 10

LEFT_HAND = 0
RIGHT_HAND = 1


class Hand:
    def __init__(self, markers, handedness):
        self.__markers__ = markers
        self.__handedness__ = handedness

    def getWrist(self):
        return [self.__markers__[0]]

    def getThumb(self):
        return self.__markers__[1:5]

    def getIndexFinger(self):
        return self.__markers__[5:9]

    def getMiddleFinger(self):
        return self.__markers__[9:13]

    def getRingFinger(self):
        return self.__markers__[13:17]

    def getPinky(self):
        return self.__markers__[17:21]

    def getHandedness(self):
        return self.__handedness__


class HandDetection:
    def __init__(self, static_image_mode=False, max_num_hands=-1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Setup media pipe
        # https://google.github.io/mediapipe/solutions/hands.html
        self.__mpHandDetection__ = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detectHands(self, frameRGB):
        dimY = len(frameRGB)
        dimX = 0 if dimY == 0 else len(frameRGB[0])
        frameDimensions = (dimX, dimY)

        hands = []

        mediaPipeHandDetectionOutput = self.__mpHandDetection__.process(frameRGB)
        multiHandLandmarks = mediaPipeHandDetectionOutput.multi_hand_landmarks
        multiHandedness = mediaPipeHandDetectionOutput.multi_handedness

        if multiHandLandmarks != None:
            for i in range(0, len(multiHandLandmarks)):
                rawLandmarks = multiHandLandmarks[i]
                rawHandedness = multiHandedness[i].classification[0].index
                markers = []
                for landmark in rawLandmarks.landmark:
                    lx = int(landmark.x * frameDimensions[0])
                    ly = int(landmark.y * frameDimensions[1])
                    markers.append((lx, ly))
                hands.append(Hand(markers, rawHandedness))

        return hands


def printHandSide(frame, hand):
    wrist = hand.getWrist()
    TEXT_FONT = cv2.FONT_HERSHEY_COMPLEX
    TEXT_COLOR = (0, 0, 255)
    TEXT_THICKNESS = 1
    TEXT_SCALE = 1
    text = None

    if hand.getHandedness() == LEFT_HAND:
        text = 'Left'
    elif hand.getHandedness() == RIGHT_HAND:
        text = 'Right'
    else:
        text = '???'

    (width, height), baseline = cv2.getTextSize(text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
    textX = int(wrist[0][0] - int(width / 2))
    textY = int(wrist[0][1] + height + baseline)
    cv2.putText(frame, text, (textX, textY), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)


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
handDetection = HandDetection(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=HAND_MIN_TRACKING_CONFIDENCE
)

print('Press "q" to quit...')

# Read and display camera capture
while True:
    _, frame = cam.read()

    if FLIP_CAMERA_FRAME_HORIZONTALY:
        frame = cv2.flip(frame, 1)

    # Capture hands and add landmarks to frame
    hands = handDetection.detectHands(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for h in hands:
        for p in h.getWrist():
            cv2.circle(frame, center=p, radius=FINGER_MARKER_RADIUS, color=FINGER_COLORS[0], thickness=cv2.FILLED)
        for p in h.getThumb():
            cv2.circle(frame, center=p, radius=FINGER_MARKER_RADIUS, color=FINGER_COLORS[1], thickness=cv2.FILLED)
        for p in h.getIndexFinger():
            cv2.circle(frame, center=p, radius=FINGER_MARKER_RADIUS, color=FINGER_COLORS[2], thickness=cv2.FILLED)
        for p in h.getMiddleFinger():
            cv2.circle(frame, center=p, radius=FINGER_MARKER_RADIUS, color=FINGER_COLORS[3], thickness=cv2.FILLED)
        for p in h.getRingFinger():
            cv2.circle(frame, center=p, radius=FINGER_MARKER_RADIUS, color=FINGER_COLORS[4], thickness=cv2.FILLED)
        for p in h.getPinky():
            cv2.circle(frame, center=p, radius=FINGER_MARKER_RADIUS, color=FINGER_COLORS[5], thickness=cv2.FILLED)
        printHandSide(frame, h)

    cv2.imshow(WINDOW_CAMERA_NAME, frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Cleanup
print('Shutting down...')
cam.release()
cv2.destroyAllWindows()
print('Application ended')
