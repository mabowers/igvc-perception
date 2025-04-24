#
# Copyright (c) 2025 Mark A. Bowers
# All rights reserved.
#
# IGVC Perception Project
# https://github.com/mabowers/igvc-perception
# ECE5532 Autonomous Vehicle Systems I, Winter 2025
# Oakland University, Rochester, MI
# Instructor: Dr. Micho Radovnikovich
#
# CMakeLists.txt file for igvc_perception ROS package
#
# This file is based on the following examples from the course materials:
# https://github.com/robustify/ece5532_project_ideas/blob/master/igvc_bag_processing/CMakeLists.txt
# https://github.com/robustify/winter2025_ece5532_examples/blob/master/opencv_example/CMakeLists.txt
# https://github.com/robustify/winter2025_ece5532_examples/blob/master/gnss_ekf_example/CMakeLists.txt
#

cmake_minimum_required(VERSION 3.0.2)
project(igvc_perception)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf2_ros
  image_geometry
  dynamic_reconfigure
  cv_bridge
)

generate_dynamic_reconfigure_options(
  cfg/CameraProcessing.cfg
  cfg/GnssEkfNode.cfg
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES pinhole_geometry_2
  DEPENDS OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# ----- Pinhole Geometry Library -----
# (Renamed 'pinhole_geometry_2' to avoid conflict with class example of the same name)

add_library(pinhole_geometry_2
  src/PinholeGeometry.cpp
)
target_link_libraries(pinhole_geometry_2
  ${catkin_LIBRARIES}
)

# ----- Camera Processing Node -----

add_executable(camera_processing_node
  src/camera_processing_node.cpp
  src/CameraProcessing.cpp
)
add_dependencies(camera_processing_node
  ${PROJECT_NAME}_gencfg
)
target_link_libraries(camera_processing_node
  pinhole_geometry_2
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

# ----- GNSS EKF Node -----

add_executable(gnss_ekf_node
  src/gnss_ekf_node.cpp
  src/GnssEkfNode.cpp
)
add_dependencies(gnss_ekf_node
  ${PROJECT_NAME}_gencfg
)
target_link_libraries(gnss_ekf_node
  ${catkin_LIBRARIES}
)