from random import randint
import cv2
import mediapipehelper as mph
import numpy as np

print(f'OpenCV version is {cv2.__version__}')

# Parameters
CAM_ID = 0
CAM_FPS = 30
CAM_RES = (640, 480)

WINDOW_CAMERA_POS = (0, 0)
WINDOW_CAMERA_NAME = 'Camera'

HAND_MIN_DETECTION_CONFIDENCE = 0.7
HAND_MIN_TRACKING_CONFIDENCE = 0.5

GAME_SPEED = 10
GAME_SPEED_INC_FACTOR = 0.1


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

    def getWidth(self):
        return self.__dim__[0]

    def getHeight(self):
        return self.__dim__[1]


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

    def getCenter(self):
        cx = self.getUpperLeftCorner()[0] + int(self.getWidth() / 2)
        cy = self.getUpperLeftCorner()[1] + int(self.getHeight() / 2)
        return (cx, cy)

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
    def __init__(self, paddle):
        self.__score__ = 0
        self.__lives__ = 3
        self.__paddle__ = paddle

    def getScore(self):
        return self.__score__

    def increaseScore(self):
        self.__score__ = self.__score__ + 1

    def getLives(self):
        return self.__lives__

    def decreaseLives(self):
        self.__lives__ = self.__lives__ - 1

    def getPaddle(self):
        return self.__paddle__


class GameEngine:
    __LEFT_WALL__ = 0
    __CEILING__ = 1
    __RIGHT_WALL__ = 2
    __FLOOR__ = 3

    def __init__(self, arena, ball, player, gameSpeed, gameSpeedIncFactor):
        self.__arena__ = arena
        self.__ball__ = ball
        self.__paddle__ = player.getPaddle()
        self.__player__ = player
        self.__isBallExploding__ = False
        self.__gameSpeed__ = gameSpeed
        self.__velocityIncFactor__ = gameSpeedIncFactor
        self.__isGameOver__ = False
        self.__resetPaddle__()
        self.__serveBall__()

    def isGameOver(self):
        return self.__isGameOver__

    def setPaddleYPosition(self, paddle, yPosition):
        if self.__isBallExploding__ or self.__isGameOver__:
            return

        arenaCeiling = self.__arena__.getUpperLeftCorner()[1]
        arenaFloor = self.__arena__.getLowerRightCorner()[1]
        newPaddleCeiling = yPosition - int(paddle.getHeight() / 2)
        newPaddleFloor = newPaddleCeiling + paddle.getHeight()

        if newPaddleCeiling < arenaCeiling:
            newPaddleCeiling = arenaCeiling
        elif newPaddleFloor > arenaFloor:
            newPaddleCeiling = arenaFloor - paddle.getHeight()

        paddle.setUpperLeftCorner((paddle.getUpperLeftCorner()[0], newPaddleCeiling))

    def tick(self):
        if self.__isBallExploding__:
            # If ball thickness is -1, we are just starting the animation.
            ballThickness = self.__ball__.getRadius() if self.__ball__.getThickness() < 0 else self.__ball__.getThickness() - 1

            if ballThickness < 1:
                self.__isBallExploding__ = False
                self.__player__.decreaseLives()

                if self.__player__.getLives() < 1:
                    self.__isGameOver__ = True
                else:
                    self.__ball__.setThickness(-1)
                    self.__resetPaddle__()
                    self.__serveBall__()
            else:
                self.__ball__.setThickness(ballThickness)

            return

        if self.__isGameOver__:
            return

        self.__ball__.move()
        collision = self.__detectCollision__()

        if collision[GameEngine.__LEFT_WALL__]:
            self.__handleLeftWallCollision__()

        if collision[GameEngine.__CEILING__]:
            self.__handleCeilingCollision__()

        if collision[GameEngine.__RIGHT_WALL__]:
            self.__handleRightWallCollision__()

        if collision[GameEngine.__FLOOR__]:
            self.__handleFloorCollision__()

    def __deflectBall__(self, paddle):
        # Increase score and game speed each time the ball is deflected
        self.__player__.increaseScore()
        vX = self.__ball__.getVelocity()[0] * -1 * (1 + self.__velocityIncFactor__)

        # Ball Y-axis velocity is based on where the ball hits the paddle relative
        # to the paddle center. The hit position is a value between [-1.0, 1.0],
        # where -1 is the paddle upper-right corner, 0 is the paddle center and
        # 1 represents the paddle lower-right corner. The new velocity is
        # REL_DIST_FROM_PADDLE_CENTER * GAME_SPEED. A hit straight at the center of
        # the paddle would result in a velocity of 0 while a hit at the top or
        # at the bottom corner would result in a velocity of -GAME_SPEED and
        # GAME_SPEED respectively.
        paddleFloor = paddle.getLowerRightCorner()[1]
        paddleCenter = paddle.getCenter()[1]
        paddleHalfHeight = float(paddleFloor - paddleCenter)
        ballCenter = self.__ball__.getCenter()[1]
        relDistFromPaddleCenter = (ballCenter - paddleCenter) / paddleHalfHeight
        vY = abs(vX) * relDistFromPaddleCenter
        self.__ball__.setVelocity((vX, vY))

    def __detectCollision__(self):
        arenaLeftWall = self.__arena__.getUpperLeftCorner()[0] + self.__player__.getPaddle().getWidth()
        arenaRightWall = self.__arena__.getLowerRightCorner()[0]
        arenaCeiling = self.__arena__.getUpperLeftCorner()[1]
        arenaFloor = self.__arena__.getLowerRightCorner()[1]

        ballCenter = self.__ball__.getCenter()
        ballRadius = self.__ball__.getRadius()
        ballLeft = ballCenter[0] - ballRadius
        ballRight = ballCenter[0] + ballRadius
        ballTop = ballCenter[1] - ballRadius
        ballBottom = ballCenter[1] + ballRadius

        isLeftCollision = True if ballLeft < arenaLeftWall else False
        isRightCollision = True if ballRight > arenaRightWall else False
        isTopCollision = True if ballTop < arenaCeiling else False
        isFloorCollision = True if ballBottom > arenaFloor else False

        return (isLeftCollision, isTopCollision, isRightCollision, isFloorCollision)

    def __handleLeftWallCollision__(self):
        arenaLeftWall = self.__arena__.getUpperLeftCorner()[0] + self.__player__.getPaddle().getWidth()
        ballX = arenaLeftWall + self.__ball__.getRadius()  # Don't display ball passed the left wall
        ballY = self.__ball__.getCenter()[1]
        self.__ball__.setCenter((ballX, ballY))

        ballCenter = self.__ball__.getCenter()[1]
        paddleCeiling = self.__player__.getPaddle().getUpperLeftCorner()[1]
        paddleFloor = self.__player__.getPaddle().getLowerRightCorner()[1]
        hasMissed = ballCenter < paddleCeiling or ballCenter > paddleFloor

        if hasMissed:
            self.__isBallExploding__ = True
        else:
            self.__deflectBall__(self.__player__.getPaddle())

    def __handleRightWallCollision__(self):
        arenaRightWall = self.__arena__.getLowerRightCorner()[0]
        ballX = arenaRightWall - self.__ball__.getRadius()  # Don't display ball passed the right wall
        ballY = self.__ball__.getCenter()[1]
        self.__ball__.setCenter((ballX, ballY))
        ballVx = self.__ball__.getVelocity()[0] * -1
        ballVy = self.__ball__.getVelocity()[1]
        self.__ball__.setVelocity((ballVx, ballVy))

    def __handleCeilingCollision__(self):
        arenaCeiling = self.__arena__.getUpperLeftCorner()[1]
        ballX = self.__ball__.getCenter()[0]
        ballY = arenaCeiling + self.__ball__.getRadius()  # Don't display ball passed the ceiling
        self.__ball__.setCenter((ballX, ballY))
        ballVx = self.__ball__.getVelocity()[0]
        ballVy = self.__ball__.getVelocity()[1] * -1
        self.__ball__.setVelocity((ballVx, ballVy))

    def __handleFloorCollision__(self):
        arenaFloor = self.__arena__.getLowerRightCorner()[1]
        ballX = self.__ball__.getCenter()[0]
        ballY = arenaFloor - self.__ball__.getRadius()  # Don't display ball passed the floor
        self.__ball__.setCenter((ballX, ballY))
        ballVx = self.__ball__.getVelocity()[0]
        ballVy = self.__ball__.getVelocity()[1] * -1
        self.__ball__.setVelocity((ballVx, ballVy))

    def __resetPaddle__(self):
        self.__player__.getPaddle().setUpperLeftCorner((
            0,
            self.__arena__.getCenter()[1] - int(self.__player__.getPaddle().getHeight() / 2)
        ))

    def __serveBall__(self):
        self.__ball__.setCenter(self.__arena__.getCenter())
        ballVx = self.__gameSpeed__
        ballVy = randint(self.__gameSpeed__ * -1, self.__gameSpeed__)  # *-1 so ball can go either up or down
        self.__ball__.setVelocity((ballVx, ballVy))


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
handDetection = mph.HandDetection(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=HAND_MIN_TRACKING_CONFIDENCE
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

    # Flip frame horizontally so left is left when you look at the screen
    frame = cv2.flip(frame, 1)

    # Try to locate the index finger position
    hands = handDetection.detectHands(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fingerCoordinates = hands[0].getLandmarks(mph.HAND_REGION_INDEX_FINGER)[3] if len(hands) else None

    # Add dark overlay
    darkOverlay = np.zeros((len(frame), len(frame[0]), 3), dtype=np.uint8)
    alpha = 0.6
    frame = cv2.addWeighted(frame, alpha, darkOverlay, 1 - alpha, 0.0)

    # Display start game / game over components based on the current game state
    if engine == None:
        hotspot = displayStartButton(frame, 0)
    elif engine.isGameOver():
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
            player = Player(paddle)
            engine = GameEngine(arena, ball, player, GAME_SPEED, GAME_SPEED_INC_FACTOR)
            hotspot = None

    # If game has been started
    if engine != None:
        if fingerCoordinates != None:
            engine.setPaddleYPosition(player.getPaddle(), fingerCoordinates[1])

        engine.tick()
        displayScoreboard(frame, player)

        # Ball and paddle are not visible when the game is over
        if not engine.isGameOver():
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
