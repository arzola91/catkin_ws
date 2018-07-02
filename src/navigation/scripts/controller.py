#! /usr/bin/env python
from __future__ import print_function

import rospy
from std_msgs.msg import Int16, Float32
import numpy as np
import roslib
roslib.load_manifest('navigation')
import sys
import math
###

class PID:

	def __init__(self,P, I, D):

		self.Kp = P
		self.Ki = I
		self.Kd = D

		self.ref    =  	0.0
		self.error  = 	0.0
		self.i_err  =   0.0
		self.d_err  =   0.0	
		self.o_time =   rospy.get_time()
		self.steer_pub = rospy.Publisher("/manual_control/steering", Int16, queue_size=1)
		self.error_sub = rospy.Subscriber("/Lateral_error",Float32,self.callback)
	
		

	def callback(self,Le):
		n_time = rospy.get_time()		#Nuevo instante de tiempo
		delta_t = n_time-self.o_time		#Diferencia entre Nuevo tiempo y viejo tiempo
		self.o_time = n_time			#Nuevo tiempo sera viejo tiempo en siguiente llamado
		
		actual_e = Le.data				#Error actual es la nueva medicion
		delta_e = actual_e-self.error			#Diferencia entre error actual y el ultimo registradp
		self.error = actual_e				#Actualizar el error. Error actual sera ahora el ultimo registrado


		err   = actual_e
		self.i_err = self.i_err+actual_e*delta_t 	
		if delta_t != 0 :		
			self.d_err = delta_e/delta_t 

		
		U = self.Kp*actual_e+self.Ki*self.i_err+self.Kd*self.d_err
		
		steer = 1450 + U

		if (steer > 2000):
        		steer = 2000
    		elif (steer < 900):
        		steer = 900
    		
		self.steer_pub.publish(steer)


def main(args):
  rospy.init_node('controller', anonymous=True)
  pid = PID(45.0,2.5,2.0)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)


