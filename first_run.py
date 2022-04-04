import cv2
import numpy as np 

hsv_min = np.array((22, 93, 0), np.uint8)
hsv_max = np.array((45, 255, 255), np.uint8)
#real width of the object (cm)
W = 6.3 

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    threshed_img = cv2.inRange(hsv, hsv_min, hsv_max)

    contours = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for c in contours:

        x, y, w, h = cv2.boundingRect(c)

        (x, y), radius = cv2.minEnclosingCircle(c)

        center = (int(x), int(y))
        radius = int(radius)
        
        w = radius
        if radius>50 :
            
            frame = cv2.circle(frame, center, radius, (255, 0, 0), 2)
            w = radius

            d = 15 #distance between object and camera (cm)
            f = (w*d)/W
            print('f = ', f)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



    cv2.imshow('mask', threshed_img)
    cv2.imshow('distance', frame)
    key = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
