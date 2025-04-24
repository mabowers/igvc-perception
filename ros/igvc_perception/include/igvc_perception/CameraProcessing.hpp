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
// Header file for the CameraProcessing class
//

#pragma once

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/LaserScan.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <fstream>

// Dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <igvc_perception/CameraProcessingConfig.h>

// Pinhole geometry
#include <igvc_perception/PinholeGeometry.hpp>

namespace igvc_perception {

  class CameraProcessing {
    public:
      CameraProcessing(ros::NodeHandle n, ros::NodeHandle pn);

    private:
      void reconfig(igvc_perception::CameraProcessingConfig& config, uint32_t level);
      void recvImage(const sensor_msgs::ImageConstPtr& msg);            // callback for camera image RX
      void recvCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg);  // callback for camera info RX
      void recvLidar(const sensor_msgs::LaserScanConstPtr& msg);        // callback for lidar scan RX

      sensor_msgs::LaserScan scan; // scan message from lidar (written by lidar callback, read by image callback)
      std::mutex scan_mutex;       // mutex to protect access to scan message

      ros::Subscriber sub_image_;        // subscriber for camera image
      ros::Subscriber sub_camera_info_;  // subscriber for camera info
      ros::Subscriber sub_lidar_;        // subscriber for lidar scan

      ros::Publisher pub_pointcloud_lines_; // publisher for point cloud message containing lines

      dynamic_reconfigure::Server<CameraProcessingConfig> srv_; // dynamic reconfigure server
      CameraProcessingConfig cfg_;  // dynamic reconfigure configuration

      std::string csv_file_path;  // path to CSV file for logging performance data
      std::ofstream csv_file;     // file stream

      std::shared_ptr<igvc_perception::PinholeGeometry> pinhole_geometry; // Pointer to PinholeGeometry object
  };

}
