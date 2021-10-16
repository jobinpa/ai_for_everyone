import cv2
import numpy as np

print(cv2.__version__)

WIDTH_SCALE_FACTOR = 4

# Saturation
satFrame = np.zeros([256, 180 * WIDTH_SCALE_FACTOR, 3], dtype=np.uint8)
for row in range(0, 256):
    for col in range(0, 180 * WIDTH_SCALE_FACTOR):
        satFrame[row, col] = [int(col / WIDTH_SCALE_FACTOR), row, 255]
satFrame = cv2.cvtColor(satFrame, cv2.COLOR_HSV2BGR)
cv2.imshow('SATURATION', satFrame)
cv2.moveWindow('SATURATION', 0, 0)

# Value
satVal = np.zeros([256, 180 * WIDTH_SCALE_FACTOR, 3], dtype=np.uint8)
for row in range(0, 256):
    for col in range(0, 180 * WIDTH_SCALE_FACTOR):
        satVal[row, col] = [int(col / WIDTH_SCALE_FACTOR), 255, row]
satVal = cv2.cvtColor(satVal, cv2.COLOR_HSV2BGR)
cv2.imshow('VALUE', satVal)
cv2.moveWindow('VALUE', 0, 0 + len(satFrame) + 50)

while True:
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
