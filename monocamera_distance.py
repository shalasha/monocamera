
import cv2
import numpy as np 

yellow_hsv_min = np.array((22, 93, 0), np.uint8)
yellow_hsv_max = np.array((45, 255, 255), np.uint8)


green_hsv_min = np.array((58,148,15), np.uint8)
green_hsv_max = np.array((99,255,255), np.uint8)

f = 442.8571428571429
W = 6.3


print('Оставить предыдущие значения переменных?')
print('y/n')
answer = input()
if answer == 'n':
    print('Введите ширину объекта:')
    W = float(input())
    print('Введите фокусное расстояние:')
    f = float(input())

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_threshed_img = cv2.inRange(hsv, yellow_hsv_min, yellow_hsv_max)
    green_threshed_img = cv2.inRange(hsv, green_hsv_min, green_hsv_max)
    
    threshed_img = yellow_threshed_img + green_threshed_img 
	
    contours = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)

        center = (int(x), int(y))
        radius = int(radius)
        
        w = radius
        if radius>50 :
            
            frame = cv2.circle(frame, center, radius, (255, 0, 0), 2)
            #cv2.putText(frame,'({},{})'.format(int(x), int(y)), (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,0), 1)
            f = 442.8571428571429
            d = (W * f) / w
            print(d)
            cv2.putText(frame,'({})'.format(d), (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,103,129), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



    cv2.imshow('mask', threshed_img)
    cv2.imshow('distance', frame)
    key = cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
