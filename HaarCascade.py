#import required modules

import cv2
import numpy as np

#load the haar cascade classifier from the xml file
cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

#function to return bounding rectangles
def detect(img, cascade):
    #detects the faces and marks the bounding rectangles
    rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=4, minSize=(50, 50), maxSize=(100, 100))
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

#auxilliary function to draw rectangles 
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

#replace './avenger.mp4' to run webcam feed
cam = cv2.VideoCapture('./avenger.mp4')

if __name__ == '__main__':
    try:
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            rects = detect(gray, cascade)
            vis = img.copy()

            draw_rects(vis, rects, (240, 32, 160))

            cv2.imshow('image', vis)

            #Press Esc to close the window
            if 0xFF & cv2.waitKey(5) == 27:
                break
        cv2.destroyAllWindows() 
        cam.release()	
    except cv2.error as e:
        exit()

        	
        
