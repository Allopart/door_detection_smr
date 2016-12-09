#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

def callback(data):

    pub = rospy.Publisher('my_image', Image, queue_size=1)
    pub.publish(data)
    
def get_image():

    rospy.init_node('door_detector', anonymous=True)
    rospy.Subscriber("camera/rgb/image_color", Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    get_image()
