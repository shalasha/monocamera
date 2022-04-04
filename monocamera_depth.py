import cv2
import numpy as np 

hsv_min = np.array((22, 93, 0), np.uint8)
hsv_max = np.array((45, 255, 255), np.uint8)
W = 6.3

cap = cv2.VideoCapture(0)
# read and scale down image
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # threshold image
    threshed_img = cv2.inRange(hsv, hsv_min, hsv_max)
    #ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 230, 255, cv2.THRESH_BINARY)

    # find contours
    contours = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)

        # get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)

        # convert all values to int
        center = (int(x), int(y))
        radius = int(radius)
        
        w = radius
        if radius>50 :
            
            #print(radius)
            frame = cv2.circle(frame, center, radius, (255, 0, 0), 2)
            #cv2.putText(frame,'({},{})'.format(int(x), int(y)), (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,0), 1)
            # Finding distance
            f = 442.8571428571429
            d = (W * f) / w
            print(d)
            cv2.putText(frame,'({})'.format(d), (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,103,129), 2)
            #w = radius
    # # Finding the Focal Length
            #d = 15
            #f = (w*d)/W
            #print('f = ', f)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



    cv2.imshow('mask', threshed_img)
    cv2.imshow('distance', frame)
    key = cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
