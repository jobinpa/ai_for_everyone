import cv2
import mediapipe as mp

print(f'OpenCV version is {cv2.__version__}')

# Parameters
CAM_ID = 0
CAM_FPS = 30
CAM_RES = (640, 480)

WINDOW_CAMERA_POS = (0, 0)
WINDOW_CAMERA_NAME = 'Camera'

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
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils

print('Press "q" to quit...')

# Read and display camera capture
while True:
    _, frame = cam.read()

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    # https://google.github.io/mediapipe/solutions/hands.html#hand-landmark-model
    myHands = []
    if results.multi_hand_landmarks != None:
        # Iterate through each hand
        for handLandMarks in results.multi_hand_landmarks:
            #mpDraw.draw_landmarks(frame, handLandMarks, mp.solutions.hands.HAND_CONNECTIONS)
            myHand = []
            myHands.append(myHand)
            for landMark in handLandMarks.landmark:
                myHand.append((int(landMark.x*frameDim[0]), int(landMark.y*frameDim[1])))

    print(myHands)
    for h in myHands:
        cv2.circle(frame, h[17], 10, (255, 0, 255), -1)
        cv2.circle(frame, h[18], 10, (255, 0, 255), -1)
        cv2.circle(frame, h[19], 10, (255, 0, 255), -1)
        cv2.circle(frame, h[20], 10, (255, 0, 255), -1)

    cv2.imshow(WINDOW_CAMERA_NAME, frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Cleanup
print('Shutting down...')
cam.release()
cv2.destroyAllWindows()
print('Application ended')
