import cv2
import mediapipehelper as mph

print(f'OpenCV version is {cv2.__version__}')

# Parameters
CAM_ID = 1
CAM_FPS = 30
CAM_RES = (1280, 1024)

FLIP_CAMERA_FRAME_HORIZONTALY = True

WINDOW_CAMERA_POS = (0, 0)
WINDOW_CAMERA_NAME = 'Camera'

FRAME_COLOR = (255, 0, 0)
DOT_RADIUS = 5
LINE_WIDTH = 2

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

# https://google.github.io/mediapipe/solutions/pose.html#cross-platform-configuration-options
poseDetection = mph.PoseDetection(
    static_image_mode=False,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Read and display camera capture
while True:
    _, frame = cam.read()

    if FLIP_CAMERA_FRAME_HORIZONTALY:
        frame = cv2.flip(frame, 1)

    pose = poseDetection.detectPose(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if pose != None:
        for m in pose.getAllMarkers():
            cv2.circle(frame, m, DOT_RADIUS, FRAME_COLOR, -1)

        cv2.line(frame, pose.getRightEar(), pose.getRightEyeOuter(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getRightEyeOuter(), pose.getRightEye(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getRightEye(), pose.getRightEyeInner(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getRightEyeInner(), pose.getNose(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getNose(), pose.getLeftEyeInner(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLeftEyeInner(), pose.getLeftEye(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLeftEye(), pose.getLeftEyeOuter(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLeftEyeOuter(), pose.getLeftEar(), FRAME_COLOR, LINE_WIDTH)

        cv2.line(frame, pose.getMouthRight(), pose.getMouthLeft(), FRAME_COLOR, LINE_WIDTH)

        cv2.line(frame, pose.getLeftShoulder(), pose.getLeftElbow(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLeftElbow(), pose.getLeftWrist(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLeftWrist(), pose.getLeftPinky(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLeftPinky(), pose.getLeftIndex(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLeftIndex(), pose.getLeftWrist(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLeftWrist(), pose.getLeftThumb(), FRAME_COLOR, LINE_WIDTH)

        cv2.line(frame, pose.getLeftHip(), pose.getLeftKnee(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLeftKnee(), pose.getLeftAnkle(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLeftAnkle(), pose.getLeftHeel(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLeftHeel(), pose.getLeftFootIndex(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLeftFootIndex(), pose.getLeftAnkle(), FRAME_COLOR, LINE_WIDTH)

        cv2.line(frame, pose.getLeftShoulder(), pose.getRightShoulder(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getRightShoulder(), pose.getRightHip(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getRightHip(), pose.getLeftHip(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLeftHip(), pose.getLeftShoulder(), FRAME_COLOR, LINE_WIDTH)

        cv2.line(frame, pose.getRightShoulder(), pose.getRightElbow(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getRightElbow(), pose.getRightWrist(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getRightWrist(), pose.getRightPinky(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getRightPinky(), pose.getRightIndex(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getRightIndex(), pose.getRightWrist(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getRightWrist(), pose.getRightThumb(), FRAME_COLOR, LINE_WIDTH)

        cv2.line(frame, pose.getRightHip(), pose.getRightKnee(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getRightKnee(), pose.getRightAnkle(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getRightAnkle(), pose.getRightHeel(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getRightHeel(), pose.getRightFootIndex(), FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getRightFootIndex(), pose.getRightAnkle(), FRAME_COLOR, LINE_WIDTH)

    cv2.imshow(WINDOW_CAMERA_NAME, frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Cleanup
print('Shutting down...')
cam.release()
cv2.destroyAllWindows()
print('Application ended')
