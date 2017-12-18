import cv2
import math
import numpy as np
video_src = 'videoData/video1final.avi'
cascade_src='classifiers/cars.xml'

video_capture = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)
count = 0
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
patience = 0


lane2_patience = 0 ##implying traffic at the red signal lane

red_signal = cv2.imread('videoData/redsignal.jpg')
green_signal = cv2.imread('videoData/greensignal.png')

while True:
    cv2.namedWindow('signal',cv2.WINDOW_NORMAL)
    cv2.imshow('signal',green_signal)
    ret, img = video_capture.read()
    height, width, channels = img.shape
    higher_limit = np.int(3*height/4)
    lower_limit = np.int(height/5)
    
    if (type(img) == type(None)):
        break
        
    crop_img = img[lower_limit:higher_limit,0:width]
    grey_crop = gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    cars_crop = car_cascade.detectMultiScale(gray, 1.1, 2)
    if(len(cars_crop)>0):
        patience = 0
        cv2.imshow('traffic', crop_img)
        print('cars in the block = ',len(cars_crop))
    else:
        patience = patience+1
        if(patience<150):
            print('patience = ',patience)
        if(patience>150 and lane2_patience==0):
            print('\n\nRED\n\n')
            currentsignal = False
            cv2.imshow('signal',red_signal)
			
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    if(len(cars_crop)>0):
        print('cars overall = ',len(cars))
    for (x,y,w,h) in cars:
        if y> lower_limit and y<=higher_limit+1:
            rect = cv2.rectangle(img,(x,y),(x+w,y+h),RED,1)
            cv2.putText(img, 'vehicle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)
            #print(y)
            if(y == higher_limit):
                cv2.line(img,(0,higher_limit+1),(width,higher_limit+1),RED,1)

    line1 = cv2.line(img,(0,higher_limit),(width,higher_limit),RED,1)
    line2 = cv2.line(img,(0,lower_limit),(width,lower_limit),GREEN,1)
    cv2.imshow('active lane', img)
    
    if cv2.waitKey(33) == 27:
        break

#cv2.destroyAllWindows()