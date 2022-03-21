# vehicle counting with python opencv

import cv2
import numpy as np


#Web camera
# Webcam = cv2.VideoCapture(0)
webcam = cv2.VideoCapture('video.mp4')

startingline = 550

min_width_rect = 100
min_height_rect = 50


#Initialize algo Background Substraction. Neglect the background and only count the vehicles(mean just focusng on the vehicles)
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def centerpoint(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx,cy
    
detect = []
offset = 6
counter = 0

while True:
    ret,frame1 = webcam.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    #blur every frame
    sub = algo.apply(blur)
    
    dilate = cv2.dilate(sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatedata = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilatedata = cv2.morphologyEx(dilatedata, cv2.MORPH_CLOSE, kernel)
    vehicle_counter,h = cv2.findContours(dilatedata, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #draw line
    cv2.line(frame1, (0,startingline), (1400,startingline), (0,255,0), 2)
    
    #track a vehicle. make rectangle around it
    for (i,c) in enumerate(vehicle_counter):
        (x,y,w,h) = cv2.boundingRect(c)
        val_counter = (w>=min_width_rect) and (h>=min_height_rect)
        if not val_counter:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(frame1, "Vehicle: "+str(counter), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)     #display counter

        
        #center/dot inside rectangle which act as counter
        center = centerpoint(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 3, (0,0,255), -1)
        
        
        for (x,y) in detect:
            if y<(startingline+offset) and y>(startingline-offset):
                counter+=1
                cv2.line(frame1, (25, startingline),(1410,startingline),(0,0,255),2)
                detect.remove((x,y))
                print("Counter = "+ str(counter))
            
    cv2.putText(frame1, "Vehicle Count: "+str(counter), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,20,255), 2)     #display counter
    cv2.putText(frame1, "Press enter to exit", (900,60), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
            
     
            
                
                
    
    
    
    # cv2.imshow('Detect',dilatedata)
    
    
    
    
    
    cv2.imshow('Highway CCTV',frame1)
    
    if cv2.waitKey(1) == 13: #enter to close
        break
    
cv2.destroyAllWindows()
webcam.release()

