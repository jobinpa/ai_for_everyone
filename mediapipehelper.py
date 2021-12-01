import mediapipe as mp

LEFT_HAND = 0
RIGHT_HAND = 1


class Face:
    def __init__(self, boxUpperLeft, boxLowerRight):
        self.__boxUpperLeft__ = boxUpperLeft
        self.__boxLowerRight__ = boxLowerRight

    def getBoxUpperLeft(self):
        return self.__boxUpperLeft__

    def getBoxLowerRight(self):
        return self.__boxLowerRight__


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

    def getAllMarkers(self):
        return list(filter(lambda x: x, self.__markers__))


class Pose:
    def __init__(self, markers):
        self.__markers__ = markers

    def getNose(self):
        return self.__markers__[0]

    def getLeftEyeInner(self):
        return self.__markers__[1]

    def getLeftEye(self):
        return self.__markers__[2]

    def getLeftEyeOuter(self):
        return self.__markers__[3]

    def getRightEyeInner(self):
        return self.__markers__[4]

    def getRightEye(self):
        return self.__markers__[5]

    def getRightEyeOuter(self):
        return self.__markers__[6]

    def getLeftEar(self):
        return self.__markers__[7]

    def getRightEar(self):
        return self.__markers__[8]

    def getMouthLeft(self):
        return self.__markers__[9]

    def getMouthRight(self):
        return self.__markers__[10]

    def getLeftShoulder(self):
        return self.__markers__[11]

    def getRightShoulder(self):
        return self.__markers__[12]

    def getLeftElbow(self):
        return self.__markers__[13]

    def getRightElbow(self):
        return self.__markers__[14]

    def getLeftWrist(self):
        return self.__markers__[15]

    def getRightWrist(self):
        return self.__markers__[16]

    def getLeftPinky(self):
        return self.__markers__[17]

    def getRightPinky(self):
        return self.__markers__[18]

    def getLeftIndex(self):
        return self.__markers__[19]

    def getRightIndex(self):
        return self.__markers__[20]

    def getLeftThumb(self):
        return self.__markers__[21]

    def getRightThumb(self):
        return self.__markers__[22]

    def getLeftHip(self):
        return self.__markers__[23]

    def getRightHip(self):
        return self.__markers__[24]

    def getLeftKnee(self):
        return self.__markers__[25]

    def getRightKnee(self):
        return self.__markers__[26]

    def getLeftAnkle(self):
        return self.__markers__[27]

    def getRightAnkle(self):
        return self.__markers__[28]

    def getLeftHeel(self):
        return self.__markers__[29]

    def getRightHeel(self):
        return self.__markers__[30]

    def getLeftFootIndex(self):
        return self.__markers__[31]

    def getRightFootIndex(self):
        return self.__markers__[32]

    def getAllMarkers(self):
        return list(self.__markers__)


class FaceDetection:
    def __init__(self, min_detection_confidence=0.5):
        # https://google.github.io/mediapipe/solutions/face_detection.html
        self.__mpFaceDetection__ = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )

    def detectFaces(self, frameRGB):
        dimY = len(frameRGB)
        dimX = 0 if dimY == 0 else len(frameRGB[0])
        mpFaceDetectionOuput = self.__mpFaceDetection__.process(frameRGB)
        faces = []
        if mpFaceDetectionOuput.detections != None:
            for f in mpFaceDetectionOuput.detections:
                bbox = f.location_data.relative_bounding_box
                topLeft = (
                    int(bbox.xmin * dimX),
                    int(bbox.ymin * dimY)
                )
                bottomRight = (
                    int((bbox.xmin + bbox.width) * dimX),
                    int((bbox.ymin + bbox.height) * dimY)
                )
                faces.append(Face(topLeft, bottomRight))
        return faces


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


class PoseDetection:
    def __init__(self, static_image_mode=False, smooth_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Setup media pipe
        # https://google.github.io/mediapipe/solutions/pose.html#cross-platform-configuration-options
        self.__mpPoseDetection__ = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detectPose(self, frameRGB):
        dimY = len(frameRGB)
        dimX = 0 if dimY == 0 else len(frameRGB[0])
        frameDimensions = (dimX, dimY)

        poseDetectionOutput = self.__mpPoseDetection__.process(frameRGB)
        poseLandmarks = poseDetectionOutput.pose_landmarks

        markers = []
        if poseLandmarks != None:
            for landmark in poseLandmarks.landmark:
                lx = int(landmark.x * frameDimensions[0])
                ly = int(landmark.y * frameDimensions[1])
                markers.append((lx, ly))

        return None if len(markers) == 0 else Pose(markers)
