import numpy as np
import cv2
cap = cv2.VideoCapture('videoData/video1final.avi')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 256;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 200
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
 
# Filter by Inertia
params.filterByInertia =True
params.minInertiaRatio = 0.5
 
detector = cv2.SimpleBlobDetector_create(params)
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)

patience = 0


red_signal = cv2.imread('videoData/redsignal.jpg')
green_signal = cv2.imread('videoData/greensignal.png')

while(1):
    cv2.namedWindow('signal',cv2.WINDOW_NORMAL)
    cv2.imshow('signal',green_signal)
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('frame',fgmask)
    height, width, channels = frame.shape
    higher_limit = np.int(3*height/4)
    lower_limit = np.int(height/5)
	

	
    # Detect blobs
    reversemask=255-fgmask
    keypoints = detector.detect(reversemask)
    if(len(keypoints)==0):
        patience = patience+1
        if(patience>150):
            currentsignal = False
            cv2.imshow('signal',red_signal)
    else:
        patience = 0
    
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), RED, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # open windows with original image, mask, res, and image with keypoints marked
    line1 = cv2.line(im_with_keypoints,(0,higher_limit),(width,higher_limit),RED,1)
    line2 = cv2.line(im_with_keypoints,(0,lower_limit),(width,lower_limit),GREEN,1)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',fgmask)
    #cv2.imshow('res',res)     
    cv2.imshow("Keypoints ", im_with_keypoints)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()