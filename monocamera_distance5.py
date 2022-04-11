#! /usr/bin/env python
# -*- coding: utf-8 -*-
#import coordinate as coor
import cv2
import numpy as np 
from PIL import Image, ImageDraw
import scipy.optimize as opt
from numpy import exp
from pyzbar.pyzbar import decode

def func(variables) :
    (x,y) = variables
    first_eq = (x-(832-280))**2 + (y-(1150-150))**2 -d_1**2
    second_eq = (x-(832-615))**2 + (y-(1150 -150))**2 -d_2**2
    return [first_eq, second_eq]


foc = 452.8032786885246
W = 13.5

d_1 = 200
d_2 = 200
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
    foc = float(input())
elif answer == '1':
    foc = 452.8032786885246
    W = 13.5

cap = cv2.VideoCapture(0)

with open('myDataFile.text') as f:
    myDataList = f.read().splitlines()

while True:
    _, frame = cap.read()
    for barcode in decode(frame):
        myData = barcode.data.decode('utf-8')    
	if myData in myDataList:
            if myData == '1':
               myOutput = '1'
               myColor = (0,255,0)
               pts = np.array([barcode.polygon],np.int32)
               pts = pts.reshape((-1,1,2))
               cv2.polylines(frame,[pts],True,myColor,5)
               pts2 = barcode.rect
               w_1 = pts2[2]
               cv2.putText(frame,myOutput,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.9,myColor,2)	    
        
               d_1 = (W * foc) / w_1
               d_1 = d_1 *10

               cv2.putText(frame,'({})'.format(d_1), (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,103,129), 2)
            if myData == '2':
                myOutput = '2'
                myColor = (0, 0, 255)
 
                pts = np.array([barcode.polygon],np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(frame,[pts],True,myColor,5)
                pts2 = barcode.rect
                w_2 = pts2[2]
                
                cv2.putText(frame,myOutput,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX,0.9,myColor,2)  
          
                d_2 = (W * foc) / w_2
                d_2 = d_2 *10

                cv2.putText(frame,'({})'.format(d_2), (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,103,129), 2)
            


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
    cv2.rectangle(img,(0,0),(861,832),(0,0,0),900)

    key = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
