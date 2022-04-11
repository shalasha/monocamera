#! /usr/bin/env python
# -*- coding: utf-8 -*-
#import coordinate as coor
import cv2
import numpy as np 
from PIL import Image, ImageDraw
import scipy.optimize as opt
from numpy import exp

def func(variables) :
    (x,y) = variables
    first_eq = (x-(832-280))**2 + (y-(1150-150))**2 -d_y**2
    second_eq = (x-(832-615))**2 + (y-(1150 -150))**2 -d_g**2
    return [first_eq, second_eq]


yellow_hsv_min = np.array((12, 157, 118), np.uint8)
yellow_hsv_max = np.array((57, 255, 255), np.uint8)

d_g=7
d_y=7



green_hsv_min = np.array((58,149,16), np.uint8)
green_hsv_max = np.array((90,255,255), np.uint8)

#! /usr/bin/env python
# -*- coding: utf-8 -*-

f = 452.8032786885246
W = 12.2

img = np.zeros ((1150, 832, 3), np.uint8) # Создать пустое изображение в оттенках серого
point_size = 1
point_color = (0, 0, 255) # BGR
thickness = 5 # может быть 0, 4, 8

print('Оставить предыдущие значения переменных? 1 - да, 2 - нет')
answer = input()
if answer == '2':
    print('Введите ширину объекта:')
    W = float(input())
    print('Введите фокусное расстояние:')
    f = float(input())
elif answer == '1':
    f = 452.8032786885246
    W = 12.2

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_threshed_img = cv2.inRange(hsv, yellow_hsv_min, yellow_hsv_max)
    green_threshed_img = cv2.inRange(hsv, green_hsv_min, green_hsv_max)
    
    mask =  yellow_threshed_img + green_threshed_img
    contours_yellow = cv2.findContours(yellow_threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours_green = cv2.findContours(green_threshed_img , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


    for c in contours_yellow:
        x, y, w, h = cv2.boundingRect(c)

		# get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)

        center = (int(x), int(y))
        radius = int(radius)
		
        w = radius
        if radius>40 :
		    
            frame = cv2.circle(frame, center, radius, (255, 0, 0), 2)
		    #cv2.putText(frame,'({},{})'.format(int(x), int(y)), (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,0), 1)
            d_y = (W * f) / w
		    
		    #print(d)
            d_y = d_y *10
            cv2.putText(frame,'({})'.format(d_y), (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,103,129), 2)
            #d_y = d_y *10

    for c in contours_green:
        x, y, w, h = cv2.boundingRect(c)

		# get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
		
        w = radius
        if radius>40 :
		    
            frame = cv2.circle(frame, center, radius, (255, 0, 0), 2)
		    #cv2.putText(frame,'({},{})'.format(int(x), int(y)), (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,0), 1)
            d_g = (W * f) / w
		    #print(d)
            d_g = d_g*10
            cv2.putText(frame,'({})'.format(d_g), (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,103,129), 2)
            #d_g = d_g*10

    #d_y, d_g = coor.coordinate(contours_yellow, contours_green, frame)
    #d_y = int(d_y)
    #d_g = int (d_g)q
    solution_x, solution_y = opt.fsolve(func, (0.1,1) )  
    solution_x=int(solution_x)
    solution_y=int(solution_y)
    print (solution_x,'and', solution_y)

     # Координаты точки, которую нужно нарисовать
    points_list = [(832-280, 1150-150), (832-615, 1150-150)]
 
    for point in points_list:
        cv2.circle(img, point, point_size, point_color, thickness)
    
    
    cv2.circle(img, (solution_x, solution_y), point_size, (0, 255, 0), thickness)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #cv2.imshow('mask_yellow', yellow_threshed_img)
    #cv2.imshow('mask_green', green_threshed_img)
    cv2.putText(img,'({},{})'.format(832-solution_x, 1155-solution_y), (solution_x+5, solution_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,0), 1)
    cv2.imshow('distance', frame)
    cv2.imshow('image', img)
    cv2.imshow('mask', mask)
    cv2.rectangle(img,(0,0),(861,832),(0,0,0),900)

    key = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
