#!/usr/bin/env python

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Image
import cv2
import numpy as np

class PointCloudToImage:
    def __init__(self):
        self.point_cloud_sub = rospy.Subscriber('/my_camera/depth/points', PointCloud2, self.point_cloud_callback)
        self.image_pub = rospy.Publisher('/depth_image', Image, queue_size=10)

    def point_cloud_callback(self, point_cloud_msg):
        # Convert PointCloud2 to numpy array
        point_cloud = list(pc2.read_points(point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True))
        points = np.array(point_cloud)

        # Assuming your camera intrinsics are known
        fx, fy = 500.0, 500.0  # Focal lengths (example values, adjust as needed)
        cx, cy = 320, 240  # Principal point (center of image, adjust as needed)
        
        height, width = 480, 640  # Image dimensions
        depth_image = np.zeros((height, width), dtype=np.float32)

        for point in points:
            x, y, z = point
            u = int(fx * x / z + cx)
            v = int(fy * y / z + cy)
            if 0 <= u < width and 0 <= v < height:
                depth_image[v, u] = z

        # Normalize and convert depth image to 8-bit for visualization
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_normalized = np.uint8(depth_image_normalized)

        # Convert OpenCV image back to ROS Image message
        try:
            ros_image = Image()
            ros_image.header = point_cloud_msg.header
            ros_image.height, ros_image.width = depth_image_normalized.shape
            ros_image.encoding = "mono8"
            ros_image.is_bigendian = 0
            ros_image.step = depth_image_normalized.shape[1]
            ros_image.data = depth_image_normalized.tobytes()
            
            self.image_pub.publish(ros_image)
        except Exception as e:
            rospy.logerr(f"Error converting OpenCV image back to ROS Image message: {e}")

if __name__ == '__main__':
    rospy.init_node('point_cloud_to_image', anonymous=True)
    pc_to_image = PointCloudToImage()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
