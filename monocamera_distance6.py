#! /usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from imutils.video import VideoStream
import argparse
import imutils
import sys
import cv2
import numpy as np 
from PIL import Image, ImageDraw
import scipy.optimize as opt
from numpy import exp
from pyzbar.pyzbar import decode
import time

def func(variables) :
    (x,y) = variables
    first_eq = (x-416)**2 + (y-128)**2 -d_1**2
    second_eq = (x-367)**2 + (y-63)**2 -d_2**2
    return [first_eq, second_eq]


foc = 943.1578947368421 #ноутбук
#foc =703.6842105263158 #телефон
W = 19

img = np.zeros ((500, 430, 3), np.uint8)  # (высота, длина) Создать пустое изображение в оттенках серого


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
    foc = 943.1578947368421
    #foc =703.6842105263158
    W = 19

cap = cv2.VideoCapture(0)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
	default="DICT_ARUCO_ORIGINAL",
	help="type of ArUCo tag to detect")
args = vars(ap.parse_args())

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
#	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
#	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
#	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
#	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print("[INFO] ArUCo tag of '{}' is not supported".format(
		args["type"]))
	sys.exit(0)

# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")


# loop over the frames from the video stream



while True:
    d_1 = -1
    d_2 = -1
    _, frame = cap.read()
    frame = imutils.resize(frame, width=1000)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,
		arucoDict, parameters=arucoParams)
    if len(corners) > 0:
		# flatten the ArUco IDs list
        ids = ids.flatten()

		# loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned
			# in top-left, top-right, bottom-right, and bottom-left
			# order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

			# convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

			# draw the bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

			# compute and draw the center (x, y)-coordinates of the
			# ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

			# draw the ArUco marker ID on the frame
            cv2.putText(frame, str(markerID),
				(topLeft[0], topLeft[1] - 15),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
            #print(markerID)
            if markerID == 66 :
                w_1 = (bottomRight[1]-topRight[1])   
                d_1 = (W * foc) / w_1
                cv2.putText(frame,'({})'.format(d_1), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,103,129), 2)

            if markerID == 65 :
                w_2 = (bottomRight[1]-topRight[1])
                d_2 = (W * foc) / w_2
                cv2.putText(frame,'({})'.format(d_2), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,103,129), 2)


               
               
    if (d_1 >0) and (d_2 >0) :
        solution_x, solution_y = opt.fsolve(func, (0.1,1) )
        dt = datetime.now().time()  
        cv2.putText(img,'Time : ' '({})'.format(dt), (5, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,103,129), 2)
        cv2.putText(img,'Coordinates (x, y) : ' '({}, {})'.format(round(solution_x, 2), round(solution_y, 2)), (5, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,103,129), 2)
        print (solution_x,'and', solution_y)

        solution_x=int(solution_x)
        solution_y=int(solution_y)
        cv2.circle(img, (solution_x, solution_y), point_size, (0, 255, 0), thickness)
        d_1 = int(d_1)
        d_2 = int(d_2)
        cv2.circle(img, (416, 128), d_1, (0, 255, 0), 1)
        cv2.circle(img, (367, 63), d_2, (0, 255, 0), 1)
        cv2.putText(img,'({},{})'.format(solution_x, solution_y), (solution_x+5, solution_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,0), 1)
    else:
        cv2.putText(img,'Tracking lost', (5, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,103,129), 2)
        
     # Координаты точки, которую нужно нарисовать
    points_list = [(416, 128 ), (367, 63)]
 
    for point in points_list:
        cv2.circle(img, point, point_size, point_color, thickness)
    
    
    
    
    
    cv2.line(img, (0,54), (200,54), (0, 255, 0), 1)
    cv2.line(img, (0,150), (400,150), (0, 255, 0), 1)
    cv2.line(img, (0,54), (0,150), (0, 255, 0), 1)
    cv2.line(img, (200,150), (200,54), (0, 255, 0), 1)
    cv2.line(img, (200,0), (400,0), (0, 255, 0), 1)
    cv2.line(img, (400,0), (400,150), (0, 255, 0), 1)
    cv2.line(img, (200,0), (200,150), (0, 255, 0), 1)
    cv2.line(img, (40,54), (40,150), (0, 255, 0), 1)
    cv2.line(img, (80,54), (80,150), (0, 255, 0), 1)
    cv2.line(img, (120,54), (120,150), (0, 255, 0), 1)
    cv2.line(img, (160,54), (160,150), (0, 255, 0), 1)
    cv2.line(img, (240,0), (240,150), (0, 255, 0), 1)
    cv2.line(img, (280,0), (280,150), (0, 255, 0), 1)
    cv2.line(img, (320,0), (320,150), (0, 255, 0), 1)
    cv2.line(img, (360,0), (360,150), (0, 255, 0), 1)
    cv2.line(img, (400,0), (400,150), (0, 255, 0), 1)
    cv2.line(img, (400,40), (200,40), (0, 255, 0), 1)
    cv2.line(img, (400,80), (0,80), (0, 255, 0), 1)
    cv2.line(img, (400,120), (0,120), (0, 255, 0), 1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #cv2.imshow('mask_yellow', yellow_threshed_img)
    #cv2.imshow('mask_green', green_threshed_img)
    
    cv2.imshow('distance', frame)
    cv2.imshow('image', img)
    cv2.rectangle(img,(0,0),(861,832),(0,0,0),900)

    key = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
