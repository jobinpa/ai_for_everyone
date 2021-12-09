import cv2
import mediapipehelper as mph

print(f'OpenCV version is {cv2.__version__}')

# Parameters
CAM_ID = 0
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
        for m in pose.getLandmarks(mph.POSE_REGION_ALL):
            cv2.circle(frame, m, DOT_RADIUS, FRAME_COLOR, -1)

        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_EAR)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_EYE_OUTER)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_EYE_OUTER)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_EYE)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_EYE)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_EYE_INNER)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_EYE_INNER)[0], pose.getLandmarks(mph.POSE_REGION_NOSE)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_NOSE)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_EYE_INNER)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_EYE_INNER)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_EYE)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_EYE)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_EYE_OUTER)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_EYE_OUTER)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_EAR)[0], FRAME_COLOR, LINE_WIDTH)

        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_MOUTH_RIGHT)[0], pose.getLandmarks(mph.POSE_REGION_MOUTH_RIGHT)[0], FRAME_COLOR, LINE_WIDTH)

        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_SHOULDER)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_ELBOW)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_ELBOW)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_WRIST)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_WRIST)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_PINKY)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_PINKY)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_INDEX)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_INDEX)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_WRIST)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_WRIST)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_THUMB)[0], FRAME_COLOR, LINE_WIDTH)

        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_HIP)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_KNEE)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_KNEE)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_ANKLE)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_ANKLE)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_HEEL)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_HEEL)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_FOOT)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_FOOT)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_ANKLE)[0], FRAME_COLOR, LINE_WIDTH)

        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_SHOULDER)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_SHOULDER)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_SHOULDER)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_HIP)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_HIP)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_HIP)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_LEFT_HIP)[0], pose.getLandmarks(mph.POSE_REGION_LEFT_SHOULDER)[0], FRAME_COLOR, LINE_WIDTH)

        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_SHOULDER)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_ELBOW)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_ELBOW)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_WRIST)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_WRIST)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_PINKY)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_PINKY)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_INDEX)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_INDEX)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_WRIST)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_WRIST)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_THUMB)[0], FRAME_COLOR, LINE_WIDTH)

        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_HIP)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_KNEE)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_KNEE)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_ANKLE)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_ANKLE)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_HEEL)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_HEEL)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_FOOT)[0], FRAME_COLOR, LINE_WIDTH)
        cv2.line(frame, pose.getLandmarks(mph.POSE_REGION_RIGHT_FOOT)[0], pose.getLandmarks(mph.POSE_REGION_RIGHT_ANKLE)[0], FRAME_COLOR, LINE_WIDTH)

    cv2.imshow(WINDOW_CAMERA_NAME, frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Cleanup
print('Shutting down...')
cam.release()
cv2.destroyAllWindows()
print('Application ended')
