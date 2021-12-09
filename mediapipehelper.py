import mediapipe as mp

HANDENESS_LEFT = 0
HANDENESS_RIGHT = 1

HAND_REGION_ALL = 0
HAND_REGION_WRIST = 1
HAND_REGION_THUMB = 2
HAND_REGION_INDEX_FINGER = 3
HAND_REGION_MIDDLE_FINGER = 4
HAND_REGION_RING_FINGER = 5
HAND_REGION_PINKY = 6

# https://github.com/tensorflow/tfjs-models/commit/838611c02f51159afdd77469ce67f0e26b7bbb23
FACEMESH_REGION_ALL = 0
FACEMESH_REGION_SILHOUETTE = 1
FACEMESH_REGION_LIPS_UPPER_OUTER = 2
FACEMESH_REGION_LIPS_LOWER_OUTER = 3
FACEMESH_REGION_LIPS_UPPER_INNER = 4
FACEMESH_REGION_LIPS_LOWER_INNER = 5
FACEMESH_REGION_RIGHT_EYE_UPPER_0 = 6
FACEMESH_REGION_RIGHT_EYE_LOWER_0 = 7
FACEMESH_REGION_RIGHT_EYE_UPPER_1 = 8
FACEMESH_REGION_RIGHT_EYE_LOWER_1 = 9
FACEMESH_REGION_RIGHT_EYE_UPPER_2 = 10
FACEMESH_REGION_RIGHT_EYE_LOWER_2 = 11
FACEMESH_REGION_RIGHT_EYE_LOWER_3 = 12
FACEMESH_REGION_RIGHT_EYEBROW_UPPER = 13
FACEMESH_REGION_RIGHT_EYEBROW_LOWER = 14
FACEMESH_REGION_LEFT_EYE_UPPER_0 = 15
FACEMESH_REGION_LEFT_EYE_LOWER_0 = 16
FACEMESH_REGION_LEFT_EYE_UPPER_1 = 17
FACEMESH_REGION_LEFT_EYE_LOWER_1 = 18
FACEMESH_REGION_LEFT_EYE_UPPER_2 = 19
FACEMESH_REGION_LEFT_EYE_LOWER_2 = 20
FACEMESH_REGION_LEFT_EYE_LOWER_3 = 21
FACEMESH_REGION_LEFT_EYEBROW_UPPER = 22
FACEMESH_REGION_LEFT_EYEBROW_LOWER = 23
FACEMESH_REGION_MIDWAY_BETWEEN_EYES = 24
FACEMESH_REGION_NOSE_TIP = 25
FACEMESH_REGION_NOSE_BOTTOM = 26
FACEMESH_REGION_NOSE_RIGHT_CORNER = 27
FACEMESH_REGION_NOSE_LEFT_CORNER = 28
FACEMESH_REGION_RIGHT_CHEEK = 29
FACEMESH_REGION_LEFT_CHEEK = 30


class Face:
    def __init__(self, boxUpperLeft, boxLowerRight):
        self.__boxUpperLeft__ = boxUpperLeft
        self.__boxLowerRight__ = boxLowerRight

    def getBoxUpperLeft(self):
        return self.__boxUpperLeft__

    def getBoxLowerRight(self):
        return self.__boxLowerRight__


class FaceMesh:
    # https://github.com/tensorflow/tfjs-models/commit/838611c02f51159afdd77469ce67f0e26b7bbb23
    __regionIndexes__ = {
        FACEMESH_REGION_ALL: range(0, 468),
        FACEMESH_REGION_SILHOUETTE: [
            10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
        ],
        FACEMESH_REGION_LIPS_UPPER_OUTER: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        FACEMESH_REGION_LIPS_LOWER_OUTER: [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        FACEMESH_REGION_LIPS_UPPER_INNER: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
        FACEMESH_REGION_LIPS_LOWER_INNER: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
        FACEMESH_REGION_RIGHT_EYE_UPPER_0: [246, 161, 160, 159, 158, 157, 173],
        FACEMESH_REGION_RIGHT_EYE_LOWER_0: [33, 7, 163, 144, 145, 153, 154, 155, 133],
        FACEMESH_REGION_RIGHT_EYE_UPPER_1: [247, 30, 29, 27, 28, 56, 190],
        FACEMESH_REGION_RIGHT_EYE_LOWER_1: [130, 25, 110, 24, 23, 22, 26, 112, 243],
        FACEMESH_REGION_RIGHT_EYE_UPPER_2: [113, 225, 224, 223, 222, 221, 189],
        FACEMESH_REGION_RIGHT_EYE_LOWER_2: [226, 31, 228, 229, 230, 231, 232, 233, 244],
        FACEMESH_REGION_RIGHT_EYE_LOWER_3: [143, 111, 117, 118, 119, 120, 121, 128, 245],
        FACEMESH_REGION_RIGHT_EYEBROW_UPPER: [156, 70, 63, 105, 66, 107, 55, 193],
        FACEMESH_REGION_RIGHT_EYEBROW_LOWER: [35, 124, 46, 53, 52, 65],
        FACEMESH_REGION_LEFT_EYE_UPPER_0: [466, 388, 387, 386, 385, 384, 398],
        FACEMESH_REGION_LEFT_EYE_LOWER_0: [263, 249, 390, 373, 374, 380, 381, 382, 362],
        FACEMESH_REGION_LEFT_EYE_UPPER_1: [467, 260, 259, 257, 258, 286, 414],
        FACEMESH_REGION_LEFT_EYE_LOWER_1: [359, 255, 339, 254, 253, 252, 256, 341, 463],
        FACEMESH_REGION_LEFT_EYE_UPPER_2: [342, 445, 444, 443, 442, 441, 413],
        FACEMESH_REGION_LEFT_EYE_LOWER_2: [446, 261, 448, 449, 450, 451, 452, 453, 464],
        FACEMESH_REGION_LEFT_EYE_LOWER_3: [372, 340, 346, 347, 348, 349, 350, 357, 465],
        FACEMESH_REGION_LEFT_EYEBROW_UPPER: [383, 300, 293, 334, 296, 336, 285, 417],
        FACEMESH_REGION_LEFT_EYEBROW_LOWER: [265, 353, 276, 283, 282, 295],
        FACEMESH_REGION_MIDWAY_BETWEEN_EYES: [168],
        FACEMESH_REGION_NOSE_TIP: [1],
        FACEMESH_REGION_NOSE_BOTTOM: [2],
        FACEMESH_REGION_NOSE_RIGHT_CORNER: [98],
        FACEMESH_REGION_NOSE_LEFT_CORNER: [327],
        FACEMESH_REGION_RIGHT_CHEEK: [205],
        FACEMESH_REGION_LEFT_CHEEK: [425]
    }

    def __init__(self, landmarks):
        self.__landmarks__ = landmarks

    def getLandmarks(self, *regions):
        landmarkIndices = set()

        if (len(regions) == 0):
            for i in FaceMesh.__regionIndexes__[FACEMESH_REGION_ALL]:
                landmarkIndices.add(i)
        else:
            for r in regions:
                for i in FaceMesh.__regionIndexes__[r]:
                    landmarkIndices.add(i)

        arr = []
        for i in set(landmarkIndices):
            arr.append(self.__landmarks__[i])
        return arr


class Hand:
    __regionIndexes__ = {
        HAND_REGION_ALL: range(0, 21),
        HAND_REGION_WRIST: [0],
        HAND_REGION_THUMB: [1, 2, 3, 4],
        HAND_REGION_INDEX_FINGER: [5, 6, 7, 8],
        HAND_REGION_MIDDLE_FINGER: [9, 10, 11, 12],
        HAND_REGION_RING_FINGER: [13, 14, 15, 16],
        HAND_REGION_PINKY: [17, 18, 19, 20]
    }

    def __init__(self, landmarks, handedness):
        self.__landmarks__ = landmarks
        self.__handedness__ = handedness

    def getLandmarks(self, *regions):
        landmarkIndices = set()

        if (len(regions) == 0):
            for i in Hand.__regionIndexes__[HAND_REGION_ALL]:
                landmarkIndices.add(i)
        else:
            for r in regions:
                for i in Hand.__regionIndexes__[r]:
                    landmarkIndices.add(i)

        arr = []
        for i in set(landmarkIndices):
            arr.append(self.__landmarks__[i])
        return arr

    def getHandedness(self):
        return self.__handedness__


class Pose:
    def __init__(self, landmarks):
        self.__landmarks__ = landmarks

    def getNose(self):
        return self.__landmarks__[0]

    def getLeftEyeInner(self):
        return self.__landmarks__[1]

    def getLeftEye(self):
        return self.__landmarks__[2]

    def getLeftEyeOuter(self):
        return self.__landmarks__[3]

    def getRightEyeInner(self):
        return self.__landmarks__[4]

    def getRightEye(self):
        return self.__landmarks__[5]

    def getRightEyeOuter(self):
        return self.__landmarks__[6]

    def getLeftEar(self):
        return self.__landmarks__[7]

    def getRightEar(self):
        return self.__landmarks__[8]

    def getMouthLeft(self):
        return self.__landmarks__[9]

    def getMouthRight(self):
        return self.__landmarks__[10]

    def getLeftShoulder(self):
        return self.__landmarks__[11]

    def getRightShoulder(self):
        return self.__landmarks__[12]

    def getLeftElbow(self):
        return self.__landmarks__[13]

    def getRightElbow(self):
        return self.__landmarks__[14]

    def getLeftWrist(self):
        return self.__landmarks__[15]

    def getRightWrist(self):
        return self.__landmarks__[16]

    def getLeftPinky(self):
        return self.__landmarks__[17]

    def getRightPinky(self):
        return self.__landmarks__[18]

    def getLeftIndex(self):
        return self.__landmarks__[19]

    def getRightIndex(self):
        return self.__landmarks__[20]

    def getLeftThumb(self):
        return self.__landmarks__[21]

    def getRightThumb(self):
        return self.__landmarks__[22]

    def getLeftHip(self):
        return self.__landmarks__[23]

    def getRightHip(self):
        return self.__landmarks__[24]

    def getLeftKnee(self):
        return self.__landmarks__[25]

    def getRightKnee(self):
        return self.__landmarks__[26]

    def getLeftAnkle(self):
        return self.__landmarks__[27]

    def getRightAnkle(self):
        return self.__landmarks__[28]

    def getLeftHeel(self):
        return self.__landmarks__[29]

    def getRightHeel(self):
        return self.__landmarks__[30]

    def getLeftFootIndex(self):
        return self.__landmarks__[31]

    def getRightFootIndex(self):
        return self.__landmarks__[32]

    def getAllLandmarks(self):
        return list(self.__landmarks__)


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


class FaceMeshDetection:
    def __init__(self, static_image_mode=False, max_num_faces=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # https://google.github.io/mediapipe/solutions/face_mesh.html
        self.__mpFaceMesh__ = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detectFaces(self, frameRGB):
        dimY = len(frameRGB)
        dimX = 0 if dimY == 0 else len(frameRGB[0])
        mpFaceMeshOuput = self.__mpFaceMesh__.process(frameRGB)
        faces = []
        if mpFaceMeshOuput.multi_face_landmarks != None:
            for f in mpFaceMeshOuput.multi_face_landmarks:
                landmarks = []
                for landmark in f.landmark:
                    lx = int(landmark.x * dimX)
                    ly = int(landmark.y * dimY)
                    landmarks.append((lx, ly))
                faces.append(FaceMesh(landmarks))
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
                landmarks = []
                for landmark in rawLandmarks.landmark:
                    lx = int(landmark.x * frameDimensions[0])
                    ly = int(landmark.y * frameDimensions[1])
                    landmarks.append((lx, ly))
                hands.append(Hand(landmarks, rawHandedness))

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

        landmarks = []
        if poseLandmarks != None:
            for landmark in poseLandmarks.landmark:
                lx = int(landmark.x * frameDimensions[0])
                ly = int(landmark.y * frameDimensions[1])
                landmarks.append((lx, ly))

        return None if len(landmarks) == 0 else Pose(landmarks)
