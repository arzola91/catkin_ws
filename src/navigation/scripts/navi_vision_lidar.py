#! /usr/bin/env python
from __future__ import print_function

import rospy
import message_filters
from std_msgs.msg import Int16, Int32, Float32
import numpy as np
###
import roslib
roslib.load_manifest('navigation')
import sys
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
import math
###

### INIT FUNCIONES IMAGEN
def region_of_interest(img,vertices):
        """
        Only keeps the region of the image defined by the polygon formed from
        'vertices'. The rest of the image is set to black
        """
        #defining a blank mask to start whit
        mask = np.zeros_like(img)

        if len(img.shape) > 2:
                channel_count = img.shape[2]
                ignore_mask_color = (255,) * channel_count
        else:
                ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)

        masked_image = cv2.bitwise_and(img, mask)

        return masked_image


def image_lanes(bin_im):                                #Encuntra el/los carril(es) en una imagen binaria
        base_img = bin_im[160:240, 0:320]               #Obtener imagen sobre cual buscar la base (solamente los primeros 50 pixeles)
        histogram = np.sum(base_img[base_img.shape[0]/2:,:], axis=0)
        ###### Buscar puntos altos de histograma
        points = []
        i = 0
        for point in histogram:
                if point > 400:
                        points += [i]
                i = i+1
        ###### dividirlos en clusters

        if (points != []):
                lane1 = [points[0]]
                lane2 = []
                j = 0

                for point in points:
                        if point-np.mean(lane1) < 25:
                                lane1 += [point]
                        else:
                                lane2 += [point]
        else:
                lane1 = []
                lane2 = []

        return lane1, lane2

def poly_for_lane(lane,im):             #out_image es im

        x_base = int(np.mean(lane))
        nwindows = 12           #Number of sections for lane
        window_height = int((240-160)/nwindows)

        nonzero = im.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 15
        minpix = 5
        x_current = x_base
        lane_inds = []

        for window in range(nwindows):
                win_y_low = im.shape[0] - (window+1)*window_height
                win_y_high = im.shape[0] - (window)*window_height
                win_x_low = x_current - margin
                win_x_high = x_current + margin
                cv2.rectangle(im,(win_x_low,win_y_low),(win_x_high,win_y_high), (0,255,0),2)

                good_inds = ((nonzeroy >= win_y_low)&(nonzeroy < win_y_high)&(nonzerox >= win_x_low)&(nonzerox < win_x_high)).nonzero()[0]

                lane_inds.append(good_inds)


                if len(good_inds) > minpix:
                        x_current = np.int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)

        lane_x = nonzerox[lane_inds]
        lane_y = nonzeroy[lane_inds]

        poly = np.polyfit(lane_y,lane_x,2)

        return poly

def center_lane_f2(xr,xl,y):

        y_c = np.zeros(240)
        x_c = np.zeros(240)

        for j in range(0,239):
                x_c[j] = (xr[j]+xl[j])/2
                y_c[j] = y[j]


        center_poly = np.polyfit(y_c,x_c,2)

        return center_poly

def center_lane_f1(x,y,s):

        y_c = np.zeros(240)
        x_c = np.zeros(240)
        d = 43
        if (s == 2):
                for j in range(0,239):

                        x_c[j] = x[j]-d
                        y_c[j] = y[j]
        elif (s == 3):
                for j in range(0,239):
                        x_c[j] = x[j]+d
                        y_c[j] = y[j]

        center_poly = np.polyfit(y_c,x_c,2)


        return center_poly


### END FUNCIONES IMAGEN

### INIT FUNCIONES LIDAR

def frontal_object(scan):
	count = scan.scan_time/scan.time_increment
	flag = 1			#0 indicates the presence of an object and 1 the absesnce of an object
	myints = np.array([345,347, 350,353, 355,358,0,3, 5,7, 10,12, 15])
	for point in myints :
		degree = np.rad2deg(scan.angle_min + scan.angle_increment*point)
		if (scan.ranges[point] < .75):  				#MODIFICAR A .5 o .6
			flag = 0

	return flag

### END FUNCIONES LIDAR


def My_callback(data1,data2):				#data1 es la imagen, data2 es el scan
		
	bridge = CvBridge()
	
	ob_f = frontal_object(data2)					#HAY UN OBJETO FRENTE A MI
	

	try:
	      cv_image = bridge.imgmsg_to_cv2(data1, "bgr8")
    	except CvBridgeError as e:
      		print(e)

        global state
    	gray = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)

	kernelSize = 5
    	blur1 = cv2.GaussianBlur(gray,(kernelSize,kernelSize),0) #Eliminar Ruido

    	minThreshold = 120
    	maxThreshold = 200
    	edge = cv2.Canny(blur1,minThreshold,maxThreshold) #Detectar esquinas

    	kernelSize = 5
    	blur2 = cv2.GaussianBlur(edge,(kernelSize,kernelSize),0) #Eliminar Ruido
        #ROI
    	lowerLeftPoint = [0, 240]
    	upperLeftPoint = [0,160]
    	upperRightPoint = [320,160]
    	lowerRightPoint = [320,240]
	
	pts = np.array([[lowerLeftPoint,upperLeftPoint,upperRightPoint,lowerRightPoint]], dtype=np.int32)
	
    	masked_image = region_of_interest(blur2,pts) ##Enmascarar area interesante 

    	pts1 = np.float32([[110,130],[195,132],[294,240],[15,240]])
    	pts3 = np.float32([[130,95],[190,95],[190,240],[130,240]])
    	M_n = cv2.getPerspectiveTransform(pts1,pts3)

    	binary_warped = cv2.warpPerspective(masked_image,M_n,(320,240))  #Transformar la imagen a vista de ave, cada pixel corresponde a .25cm apr$
    	orig_warped = cv2.warpPerspective(cv_image,M_n,(320,240))

	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	

	lane1, lane2 = image_lanes(binary_warped)

  	if   ((lane1 != []) and (lane2 != []) and (ob_f == 1) ):
		state = 1
    	elif ((lane1 != []) and (lane2 == []) and (state == 1) and (np.mean(lane1) > 160) and (ob_f == 1) ):
                state = 2
    	elif ((lane1 != []) and (lane2 == []) and (state == 1) and (np.mean(lane1) < 160) and (ob_f == 1) ):
                state = 3
    	elif ((lane1 != []) and (lane2 == []) and (state == 2) and (ob_f == 1)):
                state = 2
    	elif ((lane1 != []) and (lane2 == []) and (state == 3) and (ob_f == 1)):
                state = 3
    	elif ((lane1 == []) and (lane2 == []) and (ob_f == 1)):
		state = 4
	elif (ob_f == 0): 
                state = 5

    	if state == 1:
        	#Utilizar los dos carriles para detectar error lateral
        	left_fit = poly_for_lane(lane1,out_img)
        	right_fit = poly_for_lane(lane2,out_img)

        	ploty = np.linspace(0,binary_warped.shape[0]-1, binary_warped.shape[0])
        	left_fitx = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
        	right_fitx = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]


        	center_fit = center_lane_f2(right_fitx,left_fitx,ploty)
        	center_fitx = center_fit[0]*ploty**2+center_fit[1]*ploty+center_fit[2]

        	speed = 85*ob_f
        	#speed_pub.publish(speed)

        	center_lane_pos = center_fit[0]*240**2+center_fit[1]*240+center_fit[2]
        	Le = (center_lane_pos-160)*.5


    	elif state == 2:
        	#Utilizar un carril para detectar error lateral
        	poly_fit = poly_for_lane(lane1,out_img)
        	ploty = np.linspace(0,binary_warped.shape[0]-1, binary_warped.shape[0])
        	poly_fitx = poly_fit[0]*ploty**2+poly_fit[1]*ploty+poly_fit[2]

        	center_fit =center_lane_f1(poly_fitx,ploty,state)
        	center_fitx = center_fit[0]*ploty**2+center_fit[1]*ploty+center_fit[2]
        	speed = 70*ob_f
        	#speed_pub.publish(speed)
        	center_lane_pos = center_fit[0]*240**2+center_fit[1]*240+center_fit[2]
        	Le = (center_lane_pos-160)*.5

    	elif state == 3:
        	#Utilizar un carril para detectar error lateral
        	poly_fit = poly_for_lane(lane1,out_img)
        	ploty = np.linspace(0,binary_warped.shape[0]-1, binary_warped.shape[0])
        	poly_fitx = poly_fit[0]*ploty**2+poly_fit[1]*ploty+poly_fit[2]

        	center_fit =center_lane_f1(poly_fitx,ploty,state)
        	center_fitx = center_fit[0]*ploty**2+center_fit[1]*ploty+center_fit[2]
        	speed = 70*ob_f
        	#speed_pub.publish(speed)
        	center_lane_pos = center_fit[0]*240**2+center_fit[1]*240+center_fit[2]
        	Le = (center_lane_pos-160)*.5

    	elif state == 4:
        	#Utilizar la medicion anterior de error lateral
        	speed = 0*ob_f
        	#speed_pub.publish(speed)
        	Le = 0
        	state = 1

	elif state == 5:
		speed = 0
		Le = 0
		state = 1

	
	#if ((Le > -10) and (Le < 10) ):
	#        steer = 55*Le+1450
    	#elif (Le <= -10):
        #	steer = 900
    	#elif (Le >= 10):
        #	steer = 2000
    	
	#AGREGAR PID
	#CUANDO HAYA OBSTACULO VISION DEJA DE VER CARRILES; CORREGIR ALGO ALLI
	print("working")
	
	#steer_pub.publish(steer)
	speed_pub.publish(speed)
	error_pub.publish(Le)



rospy.init_node('el_listeno')#, anonymous=True)
data1_ = message_filters.Subscriber("/app/camera/rgb/image_raw", Image)
data2_ = message_filters.Subscriber("/scan", LaserScan)

#steer_pub = rospy.Publisher("/manual_control/steering", Int16, queue_size=1)
speed_pub = rospy.Publisher("/manual_control/speed", Int32, queue_size=1)
error_pub = rospy.Publisher("/Lateral_error",Float32,queue_size=1)
#
ts = message_filters.ApproximateTimeSynchronizer([data1_, data2_],1,10000000)   
ts.registerCallback(My_callback)

while not rospy.is_shutdown():
	rospy.spin()
	

