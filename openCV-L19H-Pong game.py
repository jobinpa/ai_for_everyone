from random import randint
import cv2
import mediapipe as mp

print(f'OpenCV version is {cv2.__version__}')

# Parameters
CAM_ID = 1
CAM_FPS = 30
CAM_RES = (640, 480)

WINDOW_CAMERA_POS = (0, 0)
WINDOW_CAMERA_NAME = 'Camera'


class Arena:
    def __init__(self, dim):
        self.__dim__ = (dim[0], dim[1])

    def getCenter(self):
        return (
            int((self.__dim__[0] - 1) / 2),
            int((self.__dim__[1] - 1) / 2),
        )

    def getUpperLeftCorner(self):
        return (0, 0)

    def getLowerRightCorner(self):
        return (self.__dim__[0] - 1, self.__dim__[1] - 1)


class Ball:
    def __init__(self, radius, color):
        self.__radius__ = radius
        self.__color__ = color
        self.__center__ = (0, 0)
        self.__velocity__ = (0, 0)
        self.__thickness__ = -1

    def getRadius(self):
        return self.__radius__

    def getColor(self):
        return self.__color__

    def getCenter(self):
        return self.__center__

    def setCenter(self, center):
        self.__center__ = (center[0], center[1])

    def getVelocity(self):
        return self.__velocity__

    def setVelocity(self, velocity):
        self.__velocity__ = (velocity[0], velocity[1])

    def move(self):
        self.__center__ = (
            round(self.__center__[0] + self.__velocity__[0]),
            round(self.__center__[1] + self.__velocity__[1])
        )

    def getThickness(self):
        return self.__thickness__

    def setThickness(self, thickness):
        self.__thickness__ = thickness


class Paddle:
    def __init__(self, width, height, color):
        self.__width__ = width
        self.__height__ = height
        self.__color__ = color
        self.__upperLeftCorner__ = (0, 0)

    def getWidth(self):
        return self.__width__

    def getHeight(self):
        return self.__height__

    def getColor(self):
        return self.__color__

    def setUpperLeftCorner(self, upperLeftCorner):
        self.__upperLeftCorner__ = (upperLeftCorner[0], upperLeftCorner[1])

    def getUpperLeftCorner(self):
        return self.__upperLeftCorner__

    def getLowerRightCorner(self):
        return (
            self.__upperLeftCorner__[0] + self.__width__,
            self.__upperLeftCorner__[1] + self.__height__
        )


class Player:
    def __init__(self):
        self.__score__ = 0
        self.__lives__ = 3

    def getScore(self):
        return self.__score__

    def increaseScore(self):
        if not self.isGameOver():
            self.__score__ = self.__score__ + 1

    def getLives(self):
        return self.__lives__

    def decreaseLives(self):
        if not self.isGameOver():
            self.__lives__ = self.__lives__ - 1

    def isGameOver(self):
        return self.__lives__ < 1


class GameEngine:
    __LEFT__ = 0
    __TOP__ = 1
    __RIGHT__ = 2
    __FLOOR__ = 3

    __MIN_VELOCITY__ = 6
    __MAX_VELOCITY__ = 10
    __VELOCITY_INC_FACTOR__ = 0.1

    def __init__(self, arena, ball, paddle, player):
        self.__arena__ = arena
        self.__ball__ = ball
        self.__paddle__ = paddle
        self.__player__ = player
        self.__isLostLifeAnimationInProgress__ = False
        ball.setCenter(arena.getCenter())
        ball.setVelocity(self.__getRandomVelocity__())
        paddle.setUpperLeftCorner((0, arena.getCenter()[1] - int(paddle.getHeight() / 2)))

    def setPaddlePosition(self, centerY):
        if self.__isLostLifeAnimationInProgress__ or self.__player__.isGameOver():
            return
        arenaTop = self.__arena__.getUpperLeftCorner()[1]
        arenaFloor = self.__arena__.getLowerRightCorner()[1]
        paddleTop = centerY - int(self.__paddle__.getHeight() / 2)

        if paddleTop < arenaTop:
            paddleTop = arenaTop
        elif paddleTop + self.__paddle__.getHeight() > arenaFloor:
            paddleTop = arenaFloor - self.__paddle__.getHeight()

        self.__paddle__.setUpperLeftCorner((self.__paddle__.getUpperLeftCorner()[0], paddleTop))

    def tick(self):
        if self.__isLostLifeAnimationInProgress__:
            ballThickness = self.__ball__.getRadius() if self.__ball__.getThickness() < 0 else self.__ball__.getThickness() - 1
            if ballThickness < 0:
                self.__isLostLifeAnimationInProgress__ = False
                self.__ball__.setVelocity(self.__getRandomVelocity__())
                self.__ball__.setThickness(-1)
                self.__ball__.setCenter(self.__arena__.getCenter())
                self.__paddle__.setUpperLeftCorner((0, arena.getCenter()[1] - int(paddle.getHeight() / 2)))
                self.__player__.decreaseLives()
            else:
                self.__ball__.setThickness(ballThickness)
            return

        if self.__player__.isGameOver():
            return

        self.__ball__.move()
        collision = self.__detectCollision__()

        if collision[GameEngine.__LEFT__]:
            self.__handleLeftCollision__()

        if collision[GameEngine.__TOP__]:
            self.__handleTopCollision__()

        if collision[GameEngine.__RIGHT__]:
            self.__handleRightCollision__()

        if collision[GameEngine.__FLOOR__]:
            self.__handleFloorCollision__()

    def __detectCollision__(self):
        arenaLeft = self.__arena__.getUpperLeftCorner()[0] + self.__paddle__.getWidth()
        arenaRight = self.__arena__.getLowerRightCorner()[0]
        arenaTop = self.__arena__.getUpperLeftCorner()[1]
        arenaFloor = self.__arena__.getLowerRightCorner()[1]

        ballCenter = self.__ball__.getCenter()
        ballRadius = self.__ball__.getRadius()
        ballLeft = ballCenter[0] - ballRadius
        ballRight = ballCenter[0] + ballRadius
        ballTop = ballCenter[1] - ballRadius
        ballFloor = ballCenter[1] + ballRadius

        isLeftCollision = True if ballLeft < arenaLeft else False
        isRightCollision = True if ballRight > arenaRight else False
        isTopCollision = True if ballTop < arenaTop else False
        isFloorCollision = True if ballFloor > arenaFloor else False

        return (isLeftCollision, isTopCollision, isRightCollision, isFloorCollision)

    def __getRandomVelocity__(self):
        vX = randint(GameEngine.__MIN_VELOCITY__, GameEngine.__MAX_VELOCITY__)
        vY = vX
        # We don't want to ball to be bouncing in diagonal, so we want
        # vX to be different than vY
        while vX == vY:
            vY = randint(GameEngine.__MIN_VELOCITY__, GameEngine.__MAX_VELOCITY__)
        vY = -1 * vY if randint(0, 1) == 1 else vY
        return (vX, vY)

    def __handleLeftCollision__(self):
        arenaLeft = self.__arena__.getUpperLeftCorner()[0] + self.__paddle__.getWidth()

        ballVelocity = ball.getVelocity()
        ballCenterY = ball.getCenter()[1]
        ballRadius = ball.getRadius()

        paddleTopY = self.__paddle__.getUpperLeftCorner()[1]
        paddleFloorY = self.__paddle__.getLowerRightCorner()[1]
        hasMissed = ballCenterY < paddleTopY or ballCenterY > paddleFloorY

        if hasMissed:
            self.__isLostLifeAnimationInProgress__ = True
        else:
            self.__player__.increaseScore()
            vX = (1 + GameEngine.__VELOCITY_INC_FACTOR__) * ballVelocity[0]
            vY = (1 + GameEngine.__VELOCITY_INC_FACTOR__) * ballVelocity[1]
            self.__ball__.setVelocity((vX * -1, vY))
            self.__ball__.setCenter((arenaLeft + ballRadius, ballCenterY))

    def __handleRightCollision__(self):
        arenaRight = self.__arena__.getLowerRightCorner()[0]
        ballVelocity = ball.getVelocity()
        ballCenterY = ball.getCenter()[1]
        ballradius = ball.getRadius()
        ball.setVelocity((ballVelocity[0] * -1, ballVelocity[1]))
        ball.setCenter((arenaRight - ballradius, ballCenterY))

    def __handleTopCollision__(self):
        arenaTop = self.__arena__.getUpperLeftCorner()[1]
        ballVelocity = ball.getVelocity()
        ballCenterX = ball.getCenter()[0]
        ballRadius = ball.getRadius()
        ball.setVelocity((ballVelocity[0], ballVelocity[1] * -1))
        ball.setCenter((ballCenterX, arenaTop + ballRadius))

    def __handleFloorCollision__(self):
        arenaFloor = self.__arena__.getLowerRightCorner()[1]
        ballVelocity = ball.getVelocity()
        ballCenterX = ball.getCenter()[0]
        ballRadius = ball.getRadius()
        ball.setVelocity((ballVelocity[0], ballVelocity[1] * -1))
        ball.setCenter((ballCenterX, arenaFloor - ballRadius))


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


def displayGameOver(frame):
    TEXT_FONT = cv2.FONT_HERSHEY_COMPLEX
    TEXT_COLOR = (255, 255, 255)
    TEXT_THICKNESS = 1
    TEXT_SCALE = 1.0
    TEXT = "GAME OVER"

    frameCenterX = int((len(frame[0]) - 1) / 2)
    frameCenterY = int((len(frame) - 1) / 2)

    (textWidth, textHeight), _ = cv2.getTextSize(TEXT, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
    textX = frameCenterX - int(textWidth / 2)
    textY = frameCenterY - int(textHeight / 2)
    cv2.putText(frame, TEXT, (textX, textY), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
    return displayStartButton(frame, int(textHeight * 1.25))


def displayScoreboard(frame, player):
    TEXT_FONT = cv2.FONT_HERSHEY_COMPLEX
    TEXT_COLOR = (255, 255, 255)
    TEXT_THICKNESS = 1
    TEXT_SCALE = 1
    BOX_PADDING_FACTOR = 0.5
    HEADER_LIVES = "LIVES"
    HEADER_SCORE = "SCORE"

    frameCenterX = int((len(frame[0]) - 1) / 2)

    (livesHeaderWidth, livesHeaderHeight), _ = cv2.getTextSize(HEADER_LIVES, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
    (livesValueWidth, livesValueHeight), _ = cv2.getTextSize(str(player.getLives()), TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
    (scoreHeaderWidth, scoreHeaderHeight), _ = cv2.getTextSize(HEADER_SCORE, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
    (scoreValueWidth, scoreValueHeight), _ = cv2.getTextSize(str(player.getScore()), TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)

    boxWidth = int(max([livesHeaderWidth, livesValueWidth, scoreHeaderWidth, scoreValueWidth]))
    boxHeight = int(max([livesHeaderHeight + livesValueHeight, scoreHeaderHeight + scoreValueHeight]))
    widthPadding = int(boxWidth * BOX_PADDING_FACTOR)
    heightPadding = int(boxHeight * BOX_PADDING_FACTOR)
    boxWidth = boxWidth + widthPadding
    boxHeight = boxHeight + heightPadding

    # Lives
    cv2.rectangle(frame, (frameCenterX - boxWidth, 0), (frameCenterX, boxHeight), (255, 255, 255), 1)
    livesHeaderX = frameCenterX - int(boxWidth / 2) - int(livesHeaderWidth / 2)
    livesHeaderY = livesHeaderHeight + int(heightPadding/3)
    cv2.putText(frame, HEADER_LIVES, (livesHeaderX, livesHeaderY), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
    livesValueX = frameCenterX - int(boxWidth / 2) - int(livesValueWidth / 2)
    livesValueY = boxHeight - int(heightPadding/3)
    cv2.putText(frame, str(player.getLives()), (livesValueX, livesValueY), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

    # Score
    cv2.rectangle(frame, (frameCenterX, 0), (frameCenterX + boxWidth, boxHeight), (255, 255, 255), 1)
    scoreHeaderX = frameCenterX + int(boxWidth / 2) - int(scoreHeaderWidth / 2)
    scoreHeaderY = scoreHeaderHeight + int(heightPadding/3)
    cv2.putText(frame, HEADER_SCORE, (scoreHeaderX, scoreHeaderY), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
    scoreValueX = frameCenterX + int(boxWidth / 2) - int(scoreValueWidth / 2)
    scoreValueY = boxHeight - int(heightPadding/3)
    cv2.putText(frame, str(player.getScore()), (scoreValueX, scoreValueY), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)


def displayStartButton(frame, yOffset):
    TEXT_FONT = cv2.FONT_HERSHEY_COMPLEX
    TEXT_COLOR = (255, 255, 255)
    TEXT_THICKNESS = 1
    TEXT_SCALE = 1
    TEXT = "START"
    BOX_PADDING_FACTOR = 0.5

    frameCenterX = int((len(frame[0]) - 1) / 2)
    frameCenterY = int((len(frame) - 1) / 2) + yOffset

    (textWidth, textHeight), _ = cv2.getTextSize(TEXT, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)

    boxWidth = textWidth
    boxHeight = textHeight
    widthPadding = int(boxWidth * BOX_PADDING_FACTOR)
    heightPadding = int(boxHeight * BOX_PADDING_FACTOR)
    boxWidth = boxWidth + widthPadding
    boxHeight = boxHeight + heightPadding

    boxUpperLeftCorner = (frameCenterX - int(boxWidth / 2), frameCenterY - int(boxHeight / 2))
    boxLowerRightCorner = (frameCenterX - int(boxWidth / 2) + boxWidth, frameCenterY - int(boxHeight / 2) + boxHeight)

    cv2.rectangle(frame, boxUpperLeftCorner, boxLowerRightCorner, (255, 255, 255), 1)
    textX = frameCenterX - int(textWidth / 2)
    textY = frameCenterY + int(textHeight / 2)
    cv2.putText(frame, TEXT, (textX, textY), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

    return (boxUpperLeftCorner, boxLowerRightCorner)


def isWithinRectangle(point, areaUpperLeftCorner, areaLowerRightCorner):
    if point[0] < areaUpperLeftCorner[0] or point[0] > areaLowerRightCorner[0]:
        return False
    if point[1] < areaUpperLeftCorner[1] or point[1] > areaLowerRightCorner[1]:
        return False
    return True


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

# Game components
arena = None
ball = None
paddle = None
player = None
engine = None
hotspot = None

# Read and display camera capture
print('Press "q" to quit...')
while True:
    _, frame = cam.read()

    # Try to locate the index finger position
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands = MPHelper.convertMultiHandLandmarksToCoordinates(
        handDetection.process(frameRGB).multi_hand_landmarks,
        frameDim)
    fingerCoordinates = MPHelper.getIndexFinger(hands[0])[3] if len(hands) else None

    # Display start game / game over components based on the current game state
    if engine == None:
        hotspot = displayStartButton(frame, 0)
    elif player.isGameOver():
        hotspot = displayGameOver(frame)

    # If both an hotspot (button) is active and a finger was detected,
    # check if the finger coordinates match the hotspot location. If so,
    # handle the current state as a button click.
    if hotspot != None and fingerCoordinates != None:
        hotspotUpperLeftCorner = hotspot[0]
        hotspotLowerRightCorner = hotspot[1]
        if isWithinRectangle(fingerCoordinates, hotspotUpperLeftCorner, hotspotLowerRightCorner):
            arena = Arena(frameDim)
            ball = Ball(radius=int(frameDim[1] * 0.04), color=(0, 0, 255))
            paddle = Paddle(width=int(frameDim[1] * 0.05), height=int(frameDim[1] * 0.2), color=(0, 255, 0))
            player = Player()
            engine = GameEngine(arena, ball, paddle, player)
            hotspot = None

    # If game has been started
    if engine != None:
        if fingerCoordinates != None:
            engine.setPaddlePosition(fingerCoordinates[1])

        engine.tick()
        displayScoreboard(frame, player)

        # Ball and paddle are not visible when the game is over
        if not player.isGameOver():
            cv2.circle(frame, ball.getCenter(), ball.getRadius(), ball.getColor(), ball.getThickness())
            # Draw dotted line
            for y in range(0, frameDim[1], 5):
                cv2.line(frame, (paddle.getWidth() - 1, y+1), (paddle.getWidth(), y + 3), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, paddle.getUpperLeftCorner(), paddle.getLowerRightCorner(), paddle.getColor(), cv2.FILLED)

    cv2.imshow(WINDOW_CAMERA_NAME, frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Cleanup
print('Shutting down...')
cam.release()
cv2.destroyAllWindows()
print('Application ended')
