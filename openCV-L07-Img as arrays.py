import cv2
import numpy as np
print(cv2.__version__)

frame = np.zeros([250, 250, 3], dtype=np.uint8)
frame[:, :125] = (0, 255, 0)
frame[:, 125:] = (0, 0, 255)

cv2.imshow('My Window', frame)

while True:
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
