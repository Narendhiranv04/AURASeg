cmake_minimum_required(VERSION 2.8.3)
project(my_package)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  pcl_ros
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs pcl_ros
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS scripts/image_converter.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

