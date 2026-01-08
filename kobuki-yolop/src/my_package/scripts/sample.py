#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2 as cv
import numpy as np
from geometry_msgs.msg import Twist
import sys

def callback(data):
    image_a = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height,data.width,-1)
    image = cv.normalize(image_a, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    cv.imshow("opencv ros image", image)
    cv.waitKey(1)  # Add this to ensure the image window is updated

if __name__ == "__main__":
    rospy.init_node('image_listener', anonymous=True)
    cv_image = rospy.Subscriber("/image_raw", Image, callback)
    rospy.spin()

