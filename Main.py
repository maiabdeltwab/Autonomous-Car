#Import libraries
from picamera.array import PiRGBArray
from scipy.stats import itemfreq
from picamera import PiCamera
from threading import Thread
import RPi.GPIO as gpio
import numpy as np
import time
import math
import cv2


        
#define pins that had been used 
ena=32    #enable for motor a 
enb=33    #enable for motor b
in1=13    #input 1
in2=15    #input 2 
in3=16    #input 3
in4=18    #input 4


#Pin initialization
def init():
    gpio.setmode(gpio.BOARD)
    gpio.setup(ena, gpio.OUT)
    gpio.setup(enb, gpio.OUT)
    gpio.setup(in1, gpio.OUT)
    gpio.setup(in2, gpio.OUT)
    gpio.setup(in3, gpio.OUT)
    gpio.setup(in4, gpio.OUT)


#define forward GPIOs state
def forward():    
    gpio.output(in1, False)
    gpio.output(in4, False)
    gpio.output(in2, True)
    gpio.output(in3, True)

#define stop GPIOs state
def stop(): 
    gpio.output(in1, False)
    gpio.output(in4, False)
    gpio.output(in2, False)
    gpio.output(in3, False)


#define set speed function that control motors speed using PWM techique
def setspeed(lspeed,rspeed):
    pwm1.ChangeDutyCycle(lspeed)
    pwm2.ChangeDutyCycle(rspeed)


init()  #set raspberry pi GPIOs by calling init() define function

pwm1= gpio.PWM(ena, 1000)  #set an intial value for PWM1 for motor a
pwm2= gpio.PWM(enb, 1000)  #set an intial value for PWM2 for motor b
pwm1.start(20)
pwm2.start(20)


minLineLength = 5 #define min line lenght 
maxLineGap = 10   #define max line gap


'''Define Lane detection part'''

'''Main lane detection method
   it take the captured image
   and return the processed image'''

def laneDetection(image):
    
   theta=0  #define theta which is used to decide the road direction 

   #convert the image from RGB scale into Gray scale to use Canny algorithm
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
   #Apply Gaussian blur to reduce noise 
   blurred = cv2.GaussianBlur(gray, (5, 5), 0)
   
   #Apply Canny edge detector 
   edged = cv2.Canny(blurred, 85, 85)
   
   #Apply hough lines detector to detect the road 
   lines = cv2.HoughLinesP(edged,1,np.pi/180,10,minLineLength,maxLineGap)

   #Check if there are lines
   if(lines !=None):
       #Calculate theta between lines 
       for x in range(0, len(lines)):
           for x1,y1,x2,y2 in lines[x]:
               cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
               theta=theta+math.atan2((y2-y1),(x2-x1))

               
   #use theta to determine the road direction 
               
   threshold=10  #define the threshold
   
   if(theta>threshold): #for left 
      forward()
      setspeed(20,90)
      time.sleep(0.1)
      print("left")
      
   if(theta<-threshold): #for right 
      forward()
      setspeed(90,20)
      time.sleep(0.1)
      print("right")
      
   if(abs(theta)<threshold): #for straight
      forward()
      setspeed(40,40)
      time.sleep(0.1)
      print ("straight")
      
   theta=0 #reset theta
   
   return image  #return the image after line detection processing 
  


'''Define sign recognition part'''


'''this method take the image
   and calculate dominant color in it using K-Means algorithm
   return the dominant color in BGR scale
'''
def get_dominant_color(image, n_colors):
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    return palette[np.argmax(itemfreq(labels)[:, -1])]



'''Main sign recognition method
   it take the captured image
   and return the processed image'''

def signRecognition(image):

    #convert the image from RGB scale into Gray scale to use Canny algorithm
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #blur it to reduce noise 
    img = cv2.medianBlur(gray, 37)

    #use hough cicrcles method to detect circles in image
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                              1, 50, param1=120, param2=40)

    #if there is any circle in image 
    if not circles is None:

        
        circles = np.uint16(np.around(circles))
        max_r, max_i = 0, 0

        #for each circle
        for i in range(len(circles[:, :, 2][0])):
            if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
                max_i = i
                max_r = circles[:, :, 2][0][i]
        x, y, r = circles[:, :, :][0][max_i]
        if y > r and x > r:
            square = image[y-r:y+r, x-r:x+r]

            #calculate dominant color in it 
            dominant_color = get_dominant_color(square, 2)

            #if the dominant color red rage>80 & the green and blue range<100   
            if dominant_color[2] > 100 and dominant_color[0] < 100 and dominant_color[1] < 100:
                print("STOP")  
                gpio.cleanup() #stop the car by clean GPIOs
                
            #if the blue rage > 80
            elif dominant_color[0] > 80:

                '''Divide the image into 3 zones
                   and calculate dominant color for each  of them'''

                #first zone 
                zone_0 = square[square.shape[0]*3//8:square.shape[0]
                                * 5//8, square.shape[1]*1//8:square.shape[1]*3//8]
                zone_0_color = get_dominant_color(zone_0, 1)

                #second zone
                zone_1 = square[square.shape[0]*1//8:square.shape[0]
                                * 3//8, square.shape[1]*3//8:square.shape[1]*5//8]
                zone_1_color = get_dominant_color(zone_1, 1)

                #third zone
                zone_2 = square[square.shape[0]*3//8:square.shape[0]
                                * 5//8, square.shape[1]*5//8:square.shape[1]*7//8]
                zone_2_color = get_dominant_color(zone_2, 1)

                #Check which sign it is using three zones
                if zone_1_color[2] < 60:
                    
                    #for left sign
                    if sum(zone_0_color) > sum(zone_2_color):
                        forward() 
                        setspeed(20,90)
                        time.sleep(0.1)
                        print("Left sign") 

                    #for right sign
                    else:
                        forward()
                        setspeed(90,20)
                        time.sleep(0.1)
                        print("Right sign")
                        
                #for forward sign
                else:
                    if sum(zone_1_color) > sum(zone_0_color) and sum(zone_1_color) > sum(zone_2_color):
                        forward()
                        setspeed(40,40)
                        time.sleep(0.1)
                        print("Forward sign")
                   
                    #for forward and left sign
                    elif sum(zone_0_color) > sum(zone_2_color):
                        forward()
                        setspeed(20,90)
                        time.sleep(0.1)
                        print("FORWARD AND LEFT sign")
                        
                    #for forward and right sign     
                    else:
                        forward()
                        setspeed(90,20)
                        time.sleep(0.1)
                        print("FORWARD AND RIGHT sign")

            #if there is no circles, do nothing            
            else:
                print("N/A")
                
        #draw the circles in captured frame 
        for i in circles[0, :]:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
            
    return image  #return the image after sign recognition processing
   




#define the pi camera and its specs
camera = PiCamera()             
camera.resolution = (640, 480)  
camera.framerate = 30           
rawCapture = PiRGBArray(camera, size=(640, 480)) 


"Define the Main Method"
def main():

   #for each frame in captured video stream
   for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

     #covert frame to image
     image = frame.array

     #Call sign recognition and lane detect functions and give them the image
     img = signRecognition(image)  
     img = laneDetection(image)
     
     #Show the processed images on the camera frame
     cv2.imshow("Camera",img)  
     key = cv2.waitKey(1) & 0xFF
     rawCapture.truncate(0)
     if key == ord("q"):
        break 


#Call the main method 
main()
