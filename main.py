import numpy as np
import glob
import cv2

img = cv2.imread('./Images/BasicBoard.png')

hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([0, 0, 255])
higher = np.array([0, 0, 255])

puckMask = cv2.inRange(hsvImg, lower, higher)


# Get rid of noise
kernel = np.ones((5,5), np.uint8)
puckMask = cv2.dilate(puckMask, None, iterations=2) 
puckMask = cv2.erode(puckMask, kernel, iterations=1)

puckMask = cv2.morphologyEx(puckMask, cv2.MORPH_OPEN, kernel)
puckMask = cv2.morphologyEx(puckMask, cv2.MORPH_CLOSE, kernel)

cnts = cv2.findContours(puckMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1] # You should get first index if using Opencv2

print("Counts", cnts)

# loop over the contours
for c in cnts:
	# compute the center of the contour
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

	# draw the contour and center of the shape on the image
	cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
	
  # cv2.circle(img, (cX, cY), 7, (0, 0, 255), -1)
	# cv2.putText(img, "center", (cX - 20, cY - 20),
  # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
  
# params = cv2.SimpleBlobDetector_Params()
# params.filterByColor = True
# params.blobColor = 255

# params.filterByArea = True
# params.minArea = 1
# params.maxArea = 100000

# params.filterByCircularity = False
# params.filterByConvexity = False
# params.filterByInertia = False
 
# detector = cv2.SimpleBlobDetector_create(params)
# keypoints = detector.detect(puckMask)
# blobPoints = cv2.drawKeypoints(puckMask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv2.imshow("Blob Points", blobPoints)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()