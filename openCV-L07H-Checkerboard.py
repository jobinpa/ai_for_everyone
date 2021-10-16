import cv2
import numpy as np

print(cv2.__version__)

CELLSIZE_PX = 50
ROW_COUNT = 8
COLUMN_COUNT = 8
COLOR_DARK = (0, 0, 0)
COLOR_LIGHT = (255, 255, 255)

checkerboard = np.zeros([CELLSIZE_PX * ROW_COUNT, CELLSIZE_PX * COLUMN_COUNT, 3], dtype=np.uint8)


def drawCell(row, col, cellColor):
    rowStart = row * CELLSIZE_PX
    rowEnd = rowStart + CELLSIZE_PX - 1
    colStart = col * CELLSIZE_PX
    colEnd = colStart + CELLSIZE_PX - 1
    checkerboard[rowStart: rowEnd, colStart:colEnd] = cellColor


def drawRow(row):
    cellColor = COLOR_LIGHT if row % 2 == 0 else COLOR_DARK
    for c in range(0, ROW_COUNT):
        drawCell(row, c, cellColor)
        cellColor = COLOR_LIGHT if cellColor == COLOR_DARK else COLOR_DARK


for r in range(0, COLUMN_COUNT):
    drawRow(r)

cv2.imshow('Checkerboard', checkerboard)

while True:
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
