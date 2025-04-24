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
// Instantiates a ROS node using the GnssEkfNode class
//
// This file is based on the following example from the course materials:
// https://github.com/robustify/winter2025_ece5532_examples/blob/master/gnss_ekf_example/src/gnss_ekf_example.cpp
//

// ROS and node class header file
#include <ros/ros.h>
#include <igvc_perception/GnssEkfNode.hpp>

int main(int argc, char** argv)
{
  // Initialize ROS and declare node handles
  ros::init(argc, argv, "gnss_ekf_node");
  ros::NodeHandle n;
  ros::NodeHandle pn("~");

  // Instantiate node class
  igvc_perception::GnssEkfNode node(n, pn);

  // Spin and process callbacks
  ros::spin();
}
