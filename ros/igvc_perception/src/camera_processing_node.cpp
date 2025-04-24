//
// Copyright (c) 2025 Mark A. Bowers
// All rights reserved.
//
// IGVC Perception Project
// https://github.com/mabowers/igvc-perception
// ECE5532 Autonomous Vehicle Systems I, Winter 2025
// Oakland University, Rochester, MI
// Instructor: Dr. Micho Radovnikovich
//
// Instantiates a ROS node using the CameraProcessing class
//
// This file is based on the following example from the course materials:
// https://github.com/robustify/winter2025_ece5532_examples/blob/master/opencv_example/src/hough_transform_node.cpp
//

#include <ros/ros.h>
#include <igvc_perception/CameraProcessing.hpp>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "camera_processing_node");
  ros::NodeHandle n;
  ros::NodeHandle pn("~");

  igvc_perception::CameraProcessing node(n, pn);

  ros::spin();
}
