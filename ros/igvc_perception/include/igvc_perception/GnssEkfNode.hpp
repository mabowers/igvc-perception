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
// Header file for the GnssEkfNode class
//
// This file is based on the following example from the course materials:
// https://github.com/robustify/winter2025_ece5532_examples/blob/master/gnss_ekf_example/src/GnssEkfExample.hpp
//

#pragma once

#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/NavSatFix.h>
#include <gps_common/conversions.h>
#include <dynamic_reconfigure/server.h>
#include <igvc_perception/GnssEkfNodeConfig.h>

#include <eigen3/Eigen/Dense>


namespace igvc_perception {

  typedef Eigen::Matrix<double, 5, 1> StateVector;
  typedef Eigen::Matrix<double, 5, 5> StateMatrix;
  enum { POS_X=0, POS_Y, HEADING, SPEED, YAW_RATE };

  class GnssEkfNode {
    public:
      GnssEkfNode(ros::NodeHandle n, ros::NodeHandle pn);

    private:
      void reconfig(GnssEkfNodeConfig& config, uint32_t level);
      void recvTwist(const geometry_msgs::TwistStampedConstPtr& msg);
      void recvFix(const sensor_msgs::NavSatFixConstPtr& msg);

      // Methods to iterate the Kalman filter
      void updateFilterGPS(const ros::Time& current_time, const Eigen::Vector2d& position);
      void updateFilterTwist(const ros::Time& current_time, const geometry_msgs::Twist& twist);

      // Methods to predict states and propagate uncertainty
      StateVector statePrediction(double dt, const StateVector& old_state);
      StateMatrix stateJacobian(double dt, const StateVector& state);
      StateMatrix covPrediction(const StateMatrix& A, const StateMatrix& Q, const StateMatrix& old_cov);

      ros::Subscriber sub_twist_;
      ros::Subscriber sub_fix_;

      dynamic_reconfigure::Server<GnssEkfNodeConfig> srv_;
      GnssEkfNodeConfig cfg_;

      tf2_ros::TransformBroadcaster broadcaster_;
      Eigen::Vector2d ref_utm_;

      // Estimate state, covariance, and current time stamp
      StateVector X_;
      StateMatrix P_;
      ros::Time estimate_stamp_;

      // Process noise covariance
      StateMatrix Q_;

      // Path for EKF position
      nav_msgs::Path path_ekf_msg;
      ros::Publisher path_ekf_pub;

      // Path for raw GPS
      nav_msgs::Path path_raw_msg;
      ros::Publisher path_raw_pub;

  };

}
