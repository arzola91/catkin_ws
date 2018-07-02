#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('navigation')
import sys
import rospy
import cv2
from std_msgs.msg import String
from std_msgs.msg import Int16
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math


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

def mean_lines(img, lines): # color, thickeness):
	
	imshape = img.shape

	ymin_global = img.shape[0]
	ymax_global = 0



	all_left_grad = []
	all_left_y = []
	all_left_x = []

	all_right_grad = []
	all_right_y = []
	all_right_x = []

	for line in lines:
		for x1,y1,x2,y2 in line:
			x_1 = float(x1)	
			x_2 = float(x2)	
			y_1 = float(y1)			
			y_2 = float(y2)		
			if (x2 != x1):			
				gradient = -(y_2-y_1) / (x_2-x_1)
			
				ymin_global = min(min(y1,y2), ymin_global)
				ymax_global = max(max(y1,y2), ymax_global)
				if (gradient > 0):
					all_left_grad += [gradient]
					all_left_y += [y_1, y_2]
					all_left_x += [x_1, x_2]
				else:
					all_right_grad += [gradient]
					all_right_y += [y_1, y_2]
					all_right_x += [x_1, x_2]

	left_mean_grad = -np.mean(all_left_grad)
	left_y_mean = np.mean(all_left_y)
	left_x_mean = np.mean(all_left_x)
	left_intercept = left_y_mean - (left_mean_grad*left_x_mean)


	right_mean_grad = -np.mean(all_right_grad)
	right_y_mean = np.mean(all_right_y)
	right_x_mean = np.mean(all_right_x)
	right_intercept = right_y_mean - (right_mean_grad*right_x_mean)



	if ((len(all_left_grad) >0) and (len(all_right_grad) >0)):
		upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
		lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
		upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
		lower_right_x =  int((ymax_global - right_intercept) / right_mean_grad)

		#cv2.line(img, (upper_left_x, ymin_global), (lower_left_x, ymax_global), color, thickeness)
		#cv2.line(img, (upper_right_x, ymin_global), (lower_right_x, ymax_global), color, thickeness)
                upper_center_x = (upper_left_x+upper_right_x)/2
		lower_center_x = (lower_left_x+lower_right_x)/2

		#cv2.line(img,(upper_center_x, ymin_global),(lower_center_x, ymax_global),[0,255,0],thickeness)
		
		left_lane = np.array([upper_left_x, ymin_global, lower_left_x, ymax_global])
		right_lane = np.array([upper_right_x, ymin_global, lower_right_x, ymax_global])	
		center_lane = np.array([upper_center_x, ymin_global, lower_center_x, ymax_global])
		
		return left_lane, right_lane, center_lane

def lane_warping(lane, tMtx):
	
	x1 = lane[0]
	y1 = lane[1]
	x2 = lane[2]
	y2 = lane[3]
	
	wx1 = (tMtx[0,0]*x1+tMtx[0,1]*y1+tMtx[0,2])/(tMtx[2,0]*x1+tMtx[2,1]*y1+tMtx[2,2])
	wy1 = (tMtx[1,0]*x1+tMtx[1,1]*y1+tMtx[1,2])/(tMtx[2,0]*x1+tMtx[2,1]*y1+tMtx[2,2])
	wx2 = (tMtx[0,0]*x2+tMtx[0,1]*y2+tMtx[0,2])/(tMtx[2,0]*x2+tMtx[2,1]*y2+tMtx[2,2])
	wy2 = (tMtx[1,0]*x2+tMtx[1,1]*y2+tMtx[1,2])/(tMtx[2,0]*x2+tMtx[2,1]*y2+tMtx[2,2])
	wlane = np.array([wx1,wy1,wx2,wy2])
	wlane = wlane.astype(int)
	return wlane
 
def draw_lane(img, lane, color, thickeness):

	cv2.line(img,(lane[0],lane[1]), (lane[2],lane[3]), color, thickeness)


def vehicle_position(lane):
	
	x1 = float(lane[0])
	y1 = float(lane[1])
	x2 = float(lane[2])
	y2 = float(lane[3])
	
	d = math.sqrt(math.pow(x2-x1,2)+math.pow(y2-y1,2))
	h = x1-x2
	ang_rad = math.asin(h/d)
	ang_deg = ang_rad*180/math.pi
	return ang_deg

def image_lanes(bin_im):                        	#Encuntra el/los carril(es) en una imagen binaria
	base_img = bin_im[160:240, 0:320]     		#Obtener imagen sobre cual buscar la base (solamente los primeros 50 pixeles)
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
	

def poly_for_lane(lane,im):		#out_image es im
	
	x_base = int(np.mean(lane))
	nwindows = 12		#Number of sections for lane
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
		d = 35
		x_c[j] = (xr[j]+xl[j])/2
		y_c[j] = y[j]


	center_poly = np.polyfit(y_c,x_c,2)

	return center_poly

def center_lane_f1(x,y,s):

	y_c = np.zeros(240)
	x_c = np.zeros(240)
	d = 37
	#if (x[479]>x[0]):		#RIGHT LANE
	if (s == 2):
		for j in range(0,239):
			
			x_c[j] = x[j]-d
			y_c[j] = y[j]
	#else:
	elif (s == 3):	
		for j in range(0,239):
			x_c[j] = x[j]+d
			y_c[j] = y[j]
	
	center_poly = np.polyfit(y_c,x_c,2)


	return center_poly
		

class image_converter:
  

  def __init__(self):
   # self.image_pub = rospy.Publisher("/Detected_lane",Image, queue_size=1)

    self.bridge = CvBridge()
    
    self.image_sub = rospy.Subscriber("/app/camera/rgb/image_raw",Image,self.callback)
    #self.object_flag_sub = rospy.Subscriber("/object_flag",Int16,self.callback)
    
    self.steer_pub = rospy.Publisher("/manual_control/steering", Int16, queue_size=1)
    self.speed_pub = rospy.Publisher("/manual_control/speed", Int32, queue_size=1)
  
  

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    #flag = 
    ####
    global state
    initial = rospy.get_time()
    gray = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY) #Transformar y mostrar a escala de grises

    kernelSize = 5
    blur1 = cv2.GaussianBlur(gray,(kernelSize,kernelSize),0) #Eliminar Ruido

    minThreshold = 120
    maxThreshold = 200
    edge = cv2.Canny(blur1,minThreshold,maxThreshold) #Detectar esquinas
    
    kernelSize = 5
    blur2 = cv2.GaussianBlur(edge,(kernelSize,kernelSize),0) #Eliminar Ruido
    
#first = [0,0]    #ROI
    lowerLeftPoint = [0, 240]
    upperLeftPoint = [0,160]
    upperRightPoint = [320,160]
    lowerRightPoint = [320,240]
#last = [640,430]
    pts = np.array([[lowerLeftPoint,upperLeftPoint,upperRightPoint,lowerRightPoint]], dtype=np.int32)
    
    masked_image = region_of_interest(blur2,pts) ##Enmascarar area interesante 

    pts1 = np.float32([[110,130],[195,132],[294,240],[15,240]])  
    pts3 = np.float32([[130,95],[190,95],[190,240],[130,240]])
    M_n = cv2.getPerspectiveTransform(pts1,pts3)	
   
    binary_warped = cv2.warpPerspective(masked_image,M_n,(320,240))  #Transformar la imagen a vista de ave, cada pixel corresponde a .25cm aprox
    orig_warped = cv2.warpPerspective(cv_image,M_n,(320,240))

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    lane1, lane2 = image_lanes(binary_warped)

    if   ((lane1 != []) and (lane2 != [])):
		state = 1
    elif ((lane1 != []) and (lane2 == []) and (state == 1) and (np.mean(lane1) > 160) ):
		state = 2
    elif ((lane1 != []) and (lane2 == []) and (state == 1) and (np.mean(lane1) < 160) ):
                state = 3
    elif ((lane1 != []) and (lane2 == []) and (state == 2)):
		state = 2
    elif ((lane1 != []) and (lane2 == []) and (state == 3)):
		state = 3
    else: #((lane1 == []) and (lane2 == []) ):
		state = 4   

    
    if state == 1:
	#Utilizar los dos carriles para detectar error lateral
	left_fit = poly_for_lane(lane1,out_img)
	right_fit = poly_for_lane(lane2,out_img)
	
	ploty = np.linspace(0,binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
	right_fitx = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
	
	
	center_fit = center_lane_f2(right_fitx,left_fitx,ploty)

	center_fitx = center_fit[0]*ploty**2+center_fit[1]*ploty+center_fit[2]
        speed = 80
        self.speed_pub.publish(speed)
	center_lane_pos = center_fit[0]*240**2+center_fit[1]*240+center_fit[2]
        Le = (center_lane_pos-160)*.5


	
    elif state == 2:
	#Utilizar un carril para detectar error lateral
	poly_fit = poly_for_lane(lane1,out_img)
	ploty = np.linspace(0,binary_warped.shape[0]-1, binary_warped.shape[0])
	poly_fitx = poly_fit[0]*ploty**2+poly_fit[1]*ploty+poly_fit[2]
#
	center_fit =center_lane_f1(poly_fitx,ploty,state)	
	center_fitx = center_fit[0]*ploty**2+center_fit[1]*ploty+center_fit[2]
	speed = 60
        self.speed_pub.publish(speed)
	center_lane_pos = center_fit[0]*240**2+center_fit[1]*240+center_fit[2]
        Le = (center_lane_pos-160)*.5

    elif state == 3:
	#Utilizar un carril para detectar error lateral
	poly_fit = poly_for_lane(lane1,out_img)
	ploty = np.linspace(0,binary_warped.shape[0]-1, binary_warped.shape[0])
	poly_fitx = poly_fit[0]*ploty**2+poly_fit[1]*ploty+poly_fit[2]

	center_fit =center_lane_f1(poly_fitx,ploty,state)	
	center_fitx = center_fit[0]*ploty**2+center_fit[1]*ploty+center_fit[2]
	speed = 60
        self.speed_pub.publish(speed)
  	center_lane_pos = center_fit[0]*240**2+center_fit[1]*240+center_fit[2]
        Le = (center_lane_pos-160)*.5
	
    elif state == 4:
	#Utilizar la medicion anterior de error lateral
	speed = 0
	self.speed_pub.publish(speed)
	Le = 0
	state = 1

	
    
    if ((Le > -10) and (Le < 10) ):
	steer = 55*Le+1450
    elif (Le <= -10):
	steer = 900
    elif (Le >= 10):
	steer = 2000
    self.steer_pub.publish(steer)
   
   # try:
    #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(centered,"bgr8"))
    # except CvBridgeError as e:
     #  print(e)
    
    ####




def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
