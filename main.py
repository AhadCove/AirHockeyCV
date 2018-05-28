import numpy as np
import glob
import cv2

img = cv2.imread('./Images/BasicBoard.png')

# We want to convert the colors to HSV to work with
hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Since we know the value won't change at all we can keep the masks the same
lower = np.array([0, 0, 255])
higher = np.array([0, 0, 255])

puckMask = cv2.inRange(hsvImg, lower, higher)

cv2.imshow("Original", img)
cv2.imshow("Puck Mask", puckMask)
cv2.waitKey(0)
cv2.destroyAllWindows()