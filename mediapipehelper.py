import mediapipe as mp

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
