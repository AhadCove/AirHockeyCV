import numpy as np
import glob
import cv2

img = cv2.imread('./Images/BasicBoard.png')


# We want to convert the colors to HSV to work with
hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of white color in HSV of puck
lower = np.array([0, 0, 255])
higher = np.array([0, 0, 255])

# Threshold the HSV image to get only puck color
puckMask = cv2.inRange(hsvImg, lower, higher)

# Reverse the image colors
# puckMask = cv2.bitwise_not(puckMask)

# Setup BlobDetector
params = cv2.SimpleBlobDetector_Params() # 2.x version should use cv2.SimpleBlobDetector()

# Threshold: min pixel intensity of interest
# params.minThreshold = 150
# params.maxThreshold = 220  

# Get white colors
params.filterByColor = True
params.blobColor = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 20
params.maxArea = 1000
	 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.10
 
# Filter by Convexity
params.filterByConvexity = False
# params.minConvexity = 0.87
	 
# Filter by Inertia
params.filterByInertia = False
# params.minInertiaRatio = 0.8

# Distance Between Blobs
# params.minDistBetweenBlobs = 0
 
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(puckMask)

# Draw detected blobs as red circles.
# Ensures the size of the circle corresponds to the size of blob
blobPoints = cv2.drawKeypoints(puckMask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Original", img)
cv2.imshow("Puck Mask", puckMask)
cv2.imshow("Blob Points", blobPoints)
cv2.waitKey(0)
cv2.destroyAllWindows()