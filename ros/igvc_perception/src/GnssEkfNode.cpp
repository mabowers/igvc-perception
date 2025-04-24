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
// Implements a GnssEkfNode ROS node that uses an Extended Kalman Filter (EKF) to combine
// GNSS and wheel speed data to produce a filtered position and heading estimate.
//
// This file is based on the following example from the course materials:
// https://github.com/robustify/winter2025_ece5532_examples/blob/master/gnss_ekf_example/src/GnssEkfExample.cpp
//

#include <igvc_perception/GnssEkfNode.hpp>

namespace igvc_perception {

  // Constructor
  GnssEkfNode::GnssEkfNode(ros::NodeHandle n, ros::NodeHandle pn)
  {
    // Load the reference coordinates and abort if they are not
    //   found on the parameter server
    double ref_lat;
    double ref_lon;
    bool found_param = true;
    found_param &= pn.getParam("ref_lat", ref_lat);
    found_param &= pn.getParam("ref_lon", ref_lon);
    if (!found_param) {
      ROS_ERROR("Could not find the reference coordinates parameters!");
      exit(1);
    }

    // Convert reference coordinates to UTM and store in 'ref_utm' variable
    std::string ref_utm_zone;
    gps_common::LLtoUTM(ref_lat, ref_lon, ref_utm_.y(), ref_utm_.x(), ref_utm_zone);

    // Subscribe to input data
    sub_fix_ = n.subscribe("gps_fix", 1, &GnssEkfNode::recvFix, this);
    sub_twist_ = n.subscribe("twist", 1, &GnssEkfNode::recvTwist, this);

    // Set up path messages
    path_ekf_pub = n.advertise<nav_msgs::Path>("path_ekf", 10);
    path_ekf_msg.header.frame_id = "map";
    path_raw_pub = n.advertise<nav_msgs::Path>("path_raw", 10);
    path_raw_msg.header.frame_id = "map";

    // Set up dynamic reconfigure server
    srv_.setCallback(boost::bind(&GnssEkfNode::reconfig, this, _1, _2));

    // Initialize Kalman filter state
    X_.setZero();
    P_.setIdentity();
  }

  void GnssEkfNode::recvTwist(const geometry_msgs::TwistStampedConstPtr& msg)
  {
    updateFilterTwist(msg->header.stamp, msg->twist);

    // Update TF transform with current estimate
    geometry_msgs::TransformStamped ekf_transform;
    ekf_transform.header.stamp = estimate_stamp_;
    ekf_transform.header.frame_id = "map";
    ekf_transform.child_frame_id = "base_footprint";
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, X_(HEADING));
    tf2::convert(q, ekf_transform.transform.rotation);
    ekf_transform.transform.translation.x = X_(0);
    ekf_transform.transform.translation.y = X_(1);
    ekf_transform.transform.translation.z = 0;
    broadcaster_.sendTransform(ekf_transform);

    // reset path if timestamp is old
    static ros::Time timestamp_prev(0);    // previous timestamp
    static ros::Time timestamp_restart(0); // first timestamp after restart
    ros::Time timestamp_now = msg->header.stamp;
    if(timestamp_now < timestamp_prev) {
      ROS_WARN("Received a message with an old timestamp!");
      path_ekf_msg.poses.clear();
      timestamp_restart = timestamp_now;
    }
    timestamp_prev = timestamp_now;

    // publish path message for EKF position
    if(timestamp_now - timestamp_restart > ros::Duration(0.5)) { // skip first 0.5s
      geometry_msgs::PoseStamped new_path_point;
      new_path_point.pose.orientation.w = 1;
      new_path_point.pose.position.x = X_(0);
      new_path_point.pose.position.y = X_(1);
      path_ekf_msg.poses.push_back(new_path_point);
      path_ekf_msg.header.stamp = timestamp_now;
      path_ekf_pub.publish(path_ekf_msg);
    }
  }

  void GnssEkfNode::recvFix(const sensor_msgs::NavSatFixConstPtr& msg)
  {
    Eigen::Vector2d current_utm;
    std::string utm_zone;
    gps_common::LLtoUTM(msg->latitude, msg->longitude, current_utm.y(), current_utm.x(), utm_zone);

    Eigen::Vector2d current_position = current_utm - ref_utm_;
    updateFilterGPS(msg->header.stamp, current_position);

    // Update TF transform with raw gps output
    geometry_msgs::TransformStamped raw_gps_transform;
    raw_gps_transform.header.stamp = msg->header.stamp;
    raw_gps_transform.header.frame_id = "map";
    raw_gps_transform.child_frame_id = "gnss";
    raw_gps_transform.transform.rotation.w = 1.0;
    raw_gps_transform.transform.translation.x = current_position.x();
    raw_gps_transform.transform.translation.y = current_position.y();
    raw_gps_transform.transform.translation.z = 0;
    broadcaster_.sendTransform(raw_gps_transform);

    // reset path if timestamp is old
    static ros::Time timestamp_prev(0); // previous timestamp
    ros::Time timestamp_now = msg->header.stamp;
    if(timestamp_now < timestamp_prev) {
      ROS_WARN("Received a message with an old timestamp!");
      path_raw_msg.poses.clear();
    }
    timestamp_prev = timestamp_now;

    // publish path message for raw GPS position
    geometry_msgs::PoseStamped new_path_point;
    new_path_point.pose.orientation.w = 1;
    new_path_point.pose.position.x = current_position.x();
    new_path_point.pose.position.y = current_position.y();
    path_raw_msg.poses.push_back(new_path_point);
    path_raw_msg.header.stamp = timestamp_now;
    path_raw_pub.publish(path_raw_msg);
  }

  StateVector GnssEkfNode::statePrediction(double dt, const StateVector& old_state) {
    // Implement state prediction step
    StateVector new_state;
    new_state(POS_X) = old_state(POS_X) + dt * old_state(SPEED) * cos(old_state(HEADING));
    new_state(POS_Y) = old_state(POS_Y) + dt * old_state(SPEED) * sin(old_state(HEADING));
    new_state(HEADING) = old_state(HEADING) + dt * old_state(YAW_RATE);
    new_state(SPEED) = old_state(SPEED);
    new_state(YAW_RATE) = old_state(YAW_RATE);
    return new_state;
  }

  StateMatrix GnssEkfNode::stateJacobian(double dt, const StateVector& state) {
    double sin_heading = sin(state(HEADING));
    double cos_heading = cos(state(HEADING));

    // Populate state Jacobian with current state values
    StateMatrix A;
    A.row(POS_X) << 1, 0, -dt * state(SPEED) * sin_heading, dt * cos_heading, 0;
    A.row(POS_Y) << 0, 1, dt * state(SPEED) * cos_heading, dt * sin_heading, 0;
    A.row(HEADING) << 0, 0, 1, 0, dt;
    A.row(SPEED) << 0, 0, 0, 1, 0;
    A.row(YAW_RATE) << 0, 0, 0, 0, 1;
    return A;
  }

  StateMatrix GnssEkfNode::covPrediction(const StateMatrix& A, const StateMatrix& Q, const StateMatrix& old_cov) {
    // Propagate covariance matrix one step
    StateMatrix new_cov;
    new_cov = A * old_cov * A.transpose() + Q;
    return new_cov;
  }

  void GnssEkfNode::updateFilterGPS(const ros::Time& current_time, const Eigen::Vector2d& position)
  {
    // Initialize state estimate directly if this is the first GPS measurement
    if (estimate_stamp_ == ros::Time(0)) {
      // start off facing south (heading = -90° = -pi/2 radians)
      X_ << position.x(), position.y(), -M_PI_2, 0.0, 0.0;
      P_.setIdentity();
      estimate_stamp_ = current_time;
      return;
    }

    // Compute amount of time to advance the state prediction
    double dt = (current_time - estimate_stamp_).toSec();

    // Propagate estimate prediction and store in predicted variables
    StateMatrix A = stateJacobian(dt, X_);
    StateVector predicted_state = statePrediction(dt, X_);
    StateMatrix predicted_cov = covPrediction(A, Q_, P_);

    // Construct C matrix for a GPS update (X and Y position measurements)
    Eigen::Matrix<double, 2, 5> C;
    C.row(0) << 1, 0, 0, 0, 0;
    C.row(1) << 0, 1, 0, 0, 0;

    // Compute expected measurement
    Eigen::Matrix<double, 2, 1> expected_meas;
    expected_meas << predicted_state(POS_X), predicted_state(POS_Y);

    // Put GPS measurements in an Eigen object
    Eigen::Matrix<double, 2, 1> real_meas = position;

    // Construct R matrix for the GPS measurements
    Eigen::Matrix<double, 2, 2> R;
    R.row(0) << cfg_.r_gps * cfg_.r_gps, 0;
    R.row(1) << 0, cfg_.r_gps * cfg_.r_gps;

    // Compute Kalman gain
    Eigen::Matrix<double, 2, 2> S;
    S = C * predicted_cov * C.transpose() + R;
    Eigen::Matrix<double, 5, 2> K;
    K = predicted_cov * C.transpose() * S.inverse();

    // Update filter estimate based on difference between actual and expected measurements
    X_ = predicted_state + K * (real_meas - expected_meas);

    // Update estimate error covariance using Kalman gain matrix
    P_ = (StateMatrix::Identity() - K * C) * predicted_cov;

    // Wrap heading estimate into the range -pi to pi
    if (X_(HEADING) > M_PI) {
      X_(HEADING) -= 2 * M_PI;
    } else if (X_(HEADING) < -M_PI) {
      X_(HEADING) += 2 * M_PI;
    }

    // Set estimate time stamp to the measurement's time
    estimate_stamp_ = current_time;
  }

  void GnssEkfNode::updateFilterTwist(const ros::Time& current_time, const geometry_msgs::Twist& twist)
  {
    if (estimate_stamp_ == ros::Time(0)) {
      ROS_WARN_THROTTLE(1.0, "Waiting for first GPS fix, ignoring this update");
      return;
    }

    // Compute amount of time to advance the state prediction
    double dt = (current_time - estimate_stamp_).toSec();

    // Propagate estimate prediction and store in predicted variables
    StateMatrix A = stateJacobian(dt, X_);
    StateVector predicted_state = statePrediction(dt, X_);
    StateMatrix predicted_cov = covPrediction(A, Q_, P_);

    // Construct C matrix for a twist update (speed and yaw rate measurement)
    Eigen::Matrix<double, 2, 5> C;
    C.row(0) << 0, 0, 0, 1, 0;
    C.row(1) << 0, 0, 0, 0, 1;

    // Compute expected measurement
    Eigen::Matrix<double, 2, 1> expected_meas;
    expected_meas << predicted_state(SPEED), predicted_state(YAW_RATE);

    // Put twist measurements in an Eigen object
    Eigen::Matrix<double, 2, 1> real_meas;
    real_meas << twist.linear.x, twist.angular.z;

    // Construct R matrix for the twist measurements
    Eigen::Matrix<double, 2, 2> R;
    R.row(0) << cfg_.r_speed * cfg_.r_speed, 0;
    R.row(1) << 0, cfg_.r_speed * cfg_.r_speed;

    // Compute Kalman gain
    Eigen::Matrix<double, 2, 2> S;
    S = C * predicted_cov * C.transpose() + R;
    Eigen::Matrix<double, 5, 2> K;
    K = predicted_cov * C.transpose() * S.inverse();

    // Update filter estimate based on difference between actual and expected measurements
    X_ = predicted_state + K * (real_meas - expected_meas);

    // Update estimate error covariance using Kalman gain matrix
    P_ = (StateMatrix::Identity() - K * C) * predicted_cov;

    // Wrap heading estimate into the range -pi to pi
    if (X_(HEADING) > M_PI) {
      X_(HEADING) -= 2 * M_PI;
    } else if (X_(HEADING) < -M_PI) {
      X_(HEADING) += 2 * M_PI;
    }

    // Set estimate time stamp to the measurement's time
    estimate_stamp_ = current_time;
  }

  void GnssEkfNode::reconfig(GnssEkfNodeConfig& config, uint32_t level)
  {
    Q_.setZero();
    Q_(POS_X, POS_X) = config.q_pos * config.q_pos;
    Q_(POS_Y, POS_Y) = config.q_pos * config.q_pos;
    Q_(HEADING, HEADING) = config.q_heading * config.q_heading;
    Q_(SPEED, SPEED) = config.q_speed * config.q_speed;
    Q_(YAW_RATE, YAW_RATE) = config.q_yaw_rate * config.q_yaw_rate;

    cfg_ = config;
  }

}