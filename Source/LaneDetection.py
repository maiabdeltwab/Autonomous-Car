from picamera.array import PiRGBArray
import RPi.GPIO as gpio
from picamera import PiCamera
import time
import cv2
import numpy as np
import math


minLineLength = 5
maxLineGap = 10
camera = PiCamera()



def laneDetection():
    
    camera.resolution = (640, 480)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(640, 480))
    theta=0
    #time.sleep(0.1)
    
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

        image = frame.array
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 85, 85)
        lines = cv2.HoughLinesP(edged,1,np.pi/180,10,minLineLength,maxLineGap)
        if(lines !=None):
           for x in range(0, len(lines)):
              for x1,y1,x2,y2 in lines[x]:
                  cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
                  theta=theta+math.atan2((y2-y1),(x2-x1))

                  
                  #print(theta)GPIO pins were connected to Raspberry for control
                  threshold=10
                  if(theta>threshold):
                    return "left"
                
                  if(theta<-threshold):
                    return "right"
                
                  if(abs(theta)<threshold):
                    return "straight"

                   
                  theta=0
                  cv2.imshow("Line Detection",image)
                  key = cv2.waitKey(1) & 0xFF
                  rawCapture.truncate(0)
                  if key == ord("q"):
                     break 


laneDetection()