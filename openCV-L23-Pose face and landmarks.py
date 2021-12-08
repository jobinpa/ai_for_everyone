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

faceMesh = mp.solutions.face_mesh.FaceMesh(False, 3, 0.5, 0.5)
mpDraw = mp.solutions.drawing_utils
drawSpecCircle=mpDraw.DrawingSpec(thickness=0, circle_radius=0, color=(255, 0, 0))
drawSpecLine=mpDraw.DrawingSpec(thickness=3, circle_radius=2, color=(0, 0, 255))

font = cv2.FONT_HERSHEY_SIMPLEX
fontSize = 0.2
fontColor = (0, 255, 255)
fontThick = 1

# Read and display camera capture
while True:
    _, frame = cam.read()

    if FLIP_CAMERA_FRAME_HORIZONTALY:
        frame = cv2.flip(frame, 1)

    frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(frameRgb)
    if results.multi_face_landmarks != None:
        for face in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, face, mp.solutions.face_mesh.FACE_CONNECTIONS, drawSpecCircle, drawSpecLine)
            for i in range(len(face.landmark)):
                lm = face.landmark[i];
                cv2.putText(frame, str(i), (int(lm.x * frameDim[0]), int(lm.y * frameDim[1])), font, fontSize, fontColor, fontThick)

    cv2.imshow(WINDOW_CAMERA_NAME, frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Cleanup
print('Shutting down...')
cam.release()
cv2.destroyAllWindows()
print('Application ended')
