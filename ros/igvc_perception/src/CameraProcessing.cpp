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
// This implements a Camera Processing class for a ROS node that:
// 1. Receives camera images and LIDAR scans
// 2. Processes camera images using two approaches
//   a. Traditional computer vision techniques + filtering based on LIDAR detections
//   b. Results from a segmentation network (YOLOv11) trained on the IGVC dataset
// 3. Outputs a point cloud of the detected lines
//
// Large portions of the CameraProcessing class are based on the following course materials:
// https://github.com/robustify/ece5532_project_ideas/blob/master/igvc_bag_processing/src/pinhole_geometry_example.cpp
// https://github.com/robustify/winter2025_ece5532_examples/blob/master/opencv_example/src/HoughTransform.cpp
//
// GitHub Copilot was used for general C++, ROS, and OpenCV reference
//

#include <igvc_perception/CameraProcessing.hpp>
#include <chrono>

namespace igvc_perception
{

// Constructor for the CameraProcessing class
CameraProcessing::CameraProcessing(ros::NodeHandle n, ros::NodeHandle pn)
{
  // Subscribe to the camera image and set callback
  sub_image_ = n.subscribe("raw_image", 1, &CameraProcessing::recvImage, this);
  srv_.setCallback(boost::bind(&CameraProcessing::reconfig, this, _1, _2));

  // Set up some windows for displaying images
  //cv::namedWindow("Raw Image", cv::WINDOW_AUTOSIZE);
  //cv::namedWindow("Blue Image", cv::WINDOW_AUTOSIZE);
  //cv::namedWindow("Thres Image", cv::WINDOW_AUTOSIZE);
  //cv::namedWindow("Erode Image", cv::WINDOW_AUTOSIZE);
  //cv::namedWindow("Dilate Image", cv::WINDOW_AUTOSIZE);
  //cv::namedWindow("Canny Image", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Lines Mask (CV)", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Lines Mask (AI)", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Lines Mask LIDAR-Filtered (CV)", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Lines Image (CV)", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Lines Image (AI)", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("LIDAR Points", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("LIDAR Box",    cv::WINDOW_AUTOSIZE);
  cv::waitKey(1);

  // Get frame IDs to give to pinhole_geometry
  std::string vehicle_frame;
  std::string camera_frame;
  std::string lidar_frame;
  pn.param("vehicle_frame", vehicle_frame, std::string("base_footprint"));
  pn.param("camera_frame", camera_frame, std::string("stereo_left"));
  pn.param("lidar_frame", lidar_frame, std::string("laser"));

  // Initialize class pointer after initializing ROS, passing the TF frame IDs to the constructor
  pinhole_geometry = std::make_shared<igvc_perception::PinholeGeometry>(vehicle_frame, camera_frame, lidar_frame);

  // The PinholeGeometry class requires TF frame transforms between footprint, camera, and lidar.
  // This loop attempts to look up the transforms once per second until it succeeds or the node is stopped
  ROS_INFO("Waiting for TF lookups to complete");
  while (ros::ok() && !pinhole_geometry->lookup_static_transforms()) {
    ros::Duration(1.0).sleep();
  }
  ROS_INFO("Done looking up TF transforms");

  // Set up a subscriber to the CameraInfo topic from the bag
  sub_camera_info_ = n.subscribe("camera_info", 1, &CameraProcessing::recvCameraInfo, this);
  sub_lidar_       = n.subscribe("scan", 1, &CameraProcessing::recvLidar, this);

  // Set up a publisher for the point cloud message
  pub_pointcloud_lines_ = n.advertise<sensor_msgs::PointCloud>("line_mask_pointcloud", 1);

  // Start a CSV file to log performance data
  // TODO: Replace hardcoded path
  csv_file_path = "/home/mark/ros/src/igvc-perception/ros/igvc_perception/performance_data_ros.csv";
  csv_file      = std::ofstream(csv_file_path, std::ios::out); // open in "out" mode to overwrite existing file
  if (csv_file.is_open()) {
    csv_file << "Frame Index,LIDAR Proc Time (ms),CV Proc Time (ms), AI Post-Proc Time (ms)\n"; // Write the header row
    csv_file.close();
  } else {
    ROS_ERROR("Failed to open CSV file for writing performance metrics!");
    return;
  }
}

// Callback for camera info RX
void CameraProcessing::recvCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg)
{
  // The CameraInfo message from the bag must be sent to the PinholeGeometry class for it to work
  pinhole_geometry->set_camera_info(*msg);
}

// Callback for lidar point cloud RX
void CameraProcessing::recvLidar(const sensor_msgs::LaserScanConstPtr& msg)
{
  ROS_INFO_STREAM("Lidar scan callback!");
  std::unique_lock<std::mutex> lock(scan_mutex);
  scan = *msg; // save lidar scan in shared variable
}

// Callback for camera image RX
void CameraProcessing::recvImage(const sensor_msgs::ImageConstPtr& msg)
{
  // measure start time of recvImage callback
  auto recvImage_start_time = std::chrono::high_resolution_clock::now();

  // Print the frame count
  int frame_index = (msg->header.seq) + cfg_.frame_jitter; // frame index is sometimes off by one?
  // ensure frame index is within bounds of pre-computed segmentation images
  if (frame_index >= 1200) {
    frame_index = 1199;
  }
  if (frame_index < 0) {
    frame_index = 0;
  }
  ROS_INFO_STREAM("Processing frame " << frame_index);

  // .--------------------.
  // |  LIDAR PROCESSING  |
  // '--------------------'

  // measure start time for LIDAR processing
  auto lidar_start_time = std::chrono::high_resolution_clock::now();

  // lock scan message mutex
  std::unique_lock<std::mutex> lock(scan_mutex);

  // we will store points as tf2::Vector3, as in pinhole_geometry_example
  std::vector<tf2::Vector3> lidar_points;

  // Convert LIDAR points to Cartesian coordinates
  // (NOTE: this conversion routine based on sample code generated by GitHub Copilot)
  for (size_t i = 0; i < scan.ranges.size(); ++i) {
    float range = scan.ranges[i];
    if (range < scan.range_min || range > scan.range_max) {
      // Skip invalid ranges
      continue;
    }

    // Calculate the angle of the current laser point
    float angle = scan.angle_min + i * scan.angle_increment;

    // Convert polar coordinates (range, angle) to Cartesian coordinates in the LIDAR frame
    float x_lidar = range * cos(angle);
    float y_lidar = range * sin(angle);
    float z_lidar = 0.0; // Assume the LIDAR is 2D (z = 0)

    // Create a tf2::Vector3 for the point
    tf2::Vector3 lidar_point(x_lidar, y_lidar, z_lidar);
    lidar_points.push_back(lidar_point);
  }

  // unlock scan mutex
  lock.unlock();

  // Image of LIDAR points projected into the camera frame
  cv::Mat lidar_image(msg->height, msg->width, CV_8UC1, cv::Scalar(0));

  // Loop through lidar_points
  for (const auto& lidar_point : lidar_points) {
    cv::Point output_pixel = pinhole_geometry->lidar_to_pixel(lidar_point);

    // Only allow pixels projected INSIDE the image bounds
    if (output_pixel.x >= 0 && output_pixel.x < lidar_image.cols &&
        output_pixel.y >= 0 && output_pixel.y < lidar_image.rows) {
      lidar_image.at<uchar>(output_pixel.y, output_pixel.x) = 255; // Set pixel to white
    }
  }

  // Show image of LIDAR points projected into the camera frame
  cv::imshow("LIDAR Points", lidar_image);
  cv::waitKey(1);

  // Draw bounding box around the LIDAR points
  cv::Mat lidar_box_image(msg->height, msg->width, CV_8UC1, cv::Scalar(0));

  // Get non-zero (active) pixels in the lidar image
  std::vector<cv::Point> non_zero_pixels;
  cv::findNonZero(lidar_image, non_zero_pixels);
  if (!non_zero_pixels.empty()) {

    // Group points into clusters based on proximity
    // (NOTE: this clustering routine is based on sample code from GitHub Copilot)
    std::vector<std::vector<cv::Point>> clusters;
    float proximity_threshold = 200.0; // Adjust this threshold as needed

    // For each non-zero pixel
    for (const auto& point : non_zero_pixels) {
        bool added_to_cluster = false;

        // Check if the point belongs to an existing cluster
        for (auto& cluster : clusters) {
            for (const auto& cluster_point : cluster) {
                float distance = cv::norm(point - cluster_point);
                if (distance < proximity_threshold) {
                    cluster.push_back(point);
                    added_to_cluster = true;
                    break;
                }
            }
            if (added_to_cluster) break;
        }

        // If the point does not belong to any cluster, create a new cluster
        if (!added_to_cluster) {
            clusters.push_back({point});
        }
    }
    // Draw bounding box for each cluster
    for (const auto& cluster : clusters) {
        if (cluster.size() > 3) { // Ignore small clusters

            // Draw a bounding box around the cluster
            cv::Rect bounding_box = cv::boundingRect(cluster);

            // Increase the width of the bounding box by 33%
            int width_increase = static_cast<int>(bounding_box.width * 0.33);
            bounding_box.x -= width_increase / 2; // Move the left edge to expand symmetrically
            bounding_box.width += width_increase;

            // Ensure the bounding box stays within the image frame
            bounding_box.x = std::max(0, bounding_box.x);
            bounding_box.width = std::min(lidar_box_image.cols - bounding_box.x, bounding_box.width);

            // Give a minimum height for bounding box and center the box vertically on the cluster
            int min_height = 300;
            if (bounding_box.height < min_height) {
                int height_diff = min_height - bounding_box.height;
                bounding_box.y -= height_diff / 2; // Move the top up
                bounding_box.height = min_height; // Set the new height
            }

            // Ensure the bounding box stays within the image frame
            bounding_box.y = std::max(0, bounding_box.y);
            bounding_box.height = std::min(lidar_box_image.rows - bounding_box.y, bounding_box.height);

            // Draw the bounding rectangle on the image
            cv::rectangle(lidar_box_image, bounding_box, cv::Scalar(255), -1); // Solid rectangle
        }
    }
  }

  // measure end time for LIDAR processing
  auto lidar_end_time = std::chrono::high_resolution_clock::now();
  auto lidar_duration = std::chrono::duration<double, std::milli>(lidar_end_time - lidar_start_time).count();
  ROS_INFO_STREAM("LIDAR processing took " << lidar_duration << " ms");

  // Display the image with the bounding box
  cv::imshow("LIDAR Box", lidar_box_image);
  cv::waitKey(1);

  // .-------------------.
  // |   CV PROCESSING   |
  // '-------------------'

  auto cv_start_time = std::chrono::high_resolution_clock::now(); // measure start time

  // Convert raw image from ROS image message into a cv::Mat
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
  cv::Mat raw_img = cv_ptr->image;

  //cv::imshow("Raw Image", raw_img);
  //cv::waitKey(1);

  // Split RGB image into its three separate channels
  std::vector<cv::Mat> split_images;
  cv::split(raw_img, split_images);

  // Extract the blue channel into its own grayscale image
  cv::Mat blue_image = split_images[0];

  //cv::imshow("Blue Image", blue_image);
  //cv::waitKey(1);

  // Apply binary threshold to create a binary image where white pixels correspond to high blue values
  cv::Mat thres_img;
  cv::threshold(blue_image, thres_img, cfg_.blue_thres, 255, cv::THRESH_BINARY);

  //cv::imshow("Thres Image", thres_img);
  //cv::waitKey(1);

  // Apply erosion to clean up noise
  cv::Mat erode_img;
  cv::erode(thres_img, erode_img, cv::Mat::ones(cfg_.erode_size, cfg_.erode_size, CV_8U));

  //cv::imshow("Erode Image", erode_img);
  //cv::waitKey(1);

  // Apply dilation to expand regions that passed the erosion filter
  cv::Mat dilate_img;
  cv::dilate(erode_img, dilate_img, cv::Mat::ones(cfg_.dilate_size, cfg_.dilate_size, CV_8U));

  //cv::imshow("Dilate Image", dilate_img);
  //cv::waitKey(1);
  cv::imshow("Lines Mask (CV)", dilate_img);
  cv::waitKey(1);

  // Clear portions of the line mask that are in the LIDAR bounding box
  cv::Mat dilate_img_filtered = dilate_img.clone();
  cv::bitwise_not(lidar_box_image, lidar_box_image);
  cv::bitwise_and(dilate_img_filtered, lidar_box_image, dilate_img_filtered);
  cv::imshow("Lines Mask LIDAR-Filtered (CV)", dilate_img_filtered);

  // Apply Canny edge detection to reduce the number of points that are passed to Hough Transform
  cv::Mat canny_img;
  cv::Canny(dilate_img_filtered, canny_img, 1, 2);
  // cv::imshow("Canny Image", canny_img);
  cv::imshow("Canny Image (CV)", canny_img);

  // Run Probabilistic Hough Transform algorithm to detect line segments
  std::vector<cv::Vec4i> line_segments;
  cv::HoughLinesP(canny_img, line_segments, cfg_.hough_rho_res, cfg_.hough_theta_res,
                  cfg_.hough_threshold, cfg_.hough_min_length, cfg_.hough_max_gap);

  // Draw detected Hough lines onto the raw image for visualization
  cv::Mat hough_img = raw_img.clone(); // Make a copy of the raw image first
  for (int i=0; i<line_segments.size(); i++){
    cv::line(hough_img, cv::Point(line_segments[i][0], line_segments[i][1]),
      cv::Point(line_segments[i][2], line_segments[i][3]), cv::Scalar(0, 0, 255));
  }

  auto cv_end_time = std::chrono::high_resolution_clock::now();
  auto cv_duration = std::chrono::duration<double, std::milli>(cv_end_time - cv_start_time).count();
  ROS_INFO_STREAM("CV processing took " << cv_duration << " ms");

  cv::imshow("Lines Image (CV)", hough_img);
  cv::waitKey(1);

  // .----------------------.
  // |  AI POST-PROCESSING  |
  // '----------------------'

  auto file_read_start_time = std::chrono::high_resolution_clock::now();

  // Open pre-processed segmentation network results
  std::ostringstream file_path;
  // TODO: Replace hardcoded path
  file_path << "/home/mark/ros/src/igvc-perception/img_lines/line_mask_"
    << std::setfill('0') << std::setw(4) << frame_index << ".png";
  cv::Mat img_line_mask = cv::imread(file_path.str(), cv::IMREAD_GRAYSCALE);

  // Check if the image was loaded correctly
  if (img_line_mask.empty()) {
    ROS_WARN_STREAM("Failed to load image: " << file_path.str());
    }

  auto file_read_end_time = std::chrono::high_resolution_clock::now();
  auto file_read_duration = std::chrono::duration<double, std::milli>(file_read_end_time - file_read_start_time).count();
  ROS_INFO_STREAM("File read took " << file_read_duration << " ms");

  auto ai_post_proc_start_time = std::chrono::high_resolution_clock::now();

  // Resize line mask from 640x640 (network size) back to 768x576 (original image size)
  cv::Mat img_line_mask_resized;
  cv::resize(img_line_mask, img_line_mask_resized, cv::Size(768, 576));
  cv::imshow("Lines Mask (AI)", img_line_mask_resized);

  // Run canny edge detection
  cv::Mat canny_img_ai;
  cv::Canny(img_line_mask_resized, canny_img_ai, 1, 2);
  cv::imshow("Canny Image (AI)", canny_img_ai);

  // Run Probabilistic Hough Transform algorithm to detect line segments
  std::vector<cv::Vec4i> line_segments_ai;
  cv::HoughLinesP(canny_img_ai, line_segments_ai, cfg_.hough_rho_res, cfg_.hough_theta_res,
                  cfg_.hough_threshold, cfg_.hough_min_length, cfg_.hough_max_gap);

  // Create a PointCloud message
  sensor_msgs::PointCloud pointcloud_msg;
  pointcloud_msg.header.frame_id = "base_footprint";
  pointcloud_msg.header.stamp = ros::Time::now();

  if (pinhole_geometry->geometry_ok()) // check that pinhole geometry is set up correctly
  {
    // Iterate through the line segments
    for (const auto& line_segment : line_segments_ai) {
      // Extract start and end points of the line segment
      cv::Point start_point(line_segment[0], line_segment[1]);
      cv::Point end_point(line_segment[2], line_segment[3]);

      // Convert start and end points to 3D points in the footprint frame
      geometry_msgs::Point32 start_footprint = pinhole_geometry->pixel_to_footprint(start_point);
      geometry_msgs::Point32 end_footprint   = pinhole_geometry->pixel_to_footprint(end_point);

      // Add the start point to the point cloud
      pointcloud_msg.points.push_back(start_footprint);

      // Interpolate points along the line
      // (NOTE: this interpolation routine based on sample code from GitHub Copilot)
      int num_interpolated_points = 10; // Number of points to add along the line
      for (int i = 1; i <= num_interpolated_points; ++i) {
        float t = static_cast<float>(i) / (num_interpolated_points + 1); // Interpolation factor
        geometry_msgs::Point32 interpolated_point;
        interpolated_point.x = start_footprint.x + t * (end_footprint.x - start_footprint.x);
        interpolated_point.y = start_footprint.y + t * (end_footprint.y - start_footprint.y);
        interpolated_point.z = start_footprint.z + t * (end_footprint.z - start_footprint.z);

        pointcloud_msg.points.push_back(interpolated_point);
      }

      // Add the end point to the point cloud
      pointcloud_msg.points.push_back(end_footprint);
    }

    // Publish the point cloud message for the lines
    pub_pointcloud_lines_.publish(pointcloud_msg);
    ROS_INFO_STREAM("Published point cloud with " << pointcloud_msg.points.size() << " points.");

    // Measure end of AI post-processing section
    auto ai_post_proc_end_time = std::chrono::high_resolution_clock::now();
    auto ai_post_proc_duration = std::chrono::duration<double, std::milli>(ai_post_proc_end_time - ai_post_proc_start_time).count();
    ROS_INFO_STREAM("AI post-processing took " << ai_post_proc_duration << " ms");

    // Blend with the raw image for visualization
    cv::Mat img_line_mask_overlayed = raw_img.clone();
    img_line_mask_overlayed.setTo(cv::Scalar(180, 105, 255), img_line_mask_resized);
    cv::Mat img_line_mask_blended;
    cv::addWeighted(raw_img, 0.5, img_line_mask_overlayed, 0.5, 0, img_line_mask_blended);

    // Draw detected Hough lines onto the raw image for visualization
    for (int i=0; i<line_segments_ai.size(); i++){
      cv::line(img_line_mask_blended, cv::Point(line_segments_ai[i][0], line_segments_ai[i][1]),
        cv::Point(line_segments_ai[i][2], line_segments_ai[i][3]), cv::Scalar(0, 0, 255));
    }

    // Display the image with the YOLO segmentation line mask
    cv::imshow("Lines Image (AI)", img_line_mask_blended);
    cv::waitKey(1);

    // Append performance data to CSV file
    csv_file.open(csv_file_path, std::ios::app); // open in "app" mode to append to existing file
    if (csv_file.is_open()) {
      csv_file << frame_index << "," << lidar_duration << "," << cv_duration << "," << ai_post_proc_duration << "\n"; // Write the data
      csv_file.close();
    } else {
      ROS_ERROR("Failed to open CSV file for writing performance metrics!");
    }

  }
  else
  {
    ROS_WARN_STREAM("Pinhole geometry is not set up yet. Skipping point cloud for line.");
  }

  // Measure end of image handler
  auto recvImage_end_time = std::chrono::high_resolution_clock::now();
  auto recvImage_duration = std::chrono::duration<double, std::milli>(recvImage_end_time - recvImage_start_time).count();
  ROS_INFO_STREAM("recvImage handler took " << recvImage_duration << " ms");

  if (recvImage_duration > 100) {
    ROS_WARN_STREAM("recvImage handler took longer than 100 ms!");
  }

}

void CameraProcessing::reconfig(igvc_perception::CameraProcessingConfig& config, uint32_t level)
{

  // Force erosion and dilation filter sizes to be an odd number
  if ((config.erode_size % 2) == 0) {
    config.erode_size--;
  }

  if ((config.dilate_size % 2) == 0) {
    config.dilate_size--;
  }

  cfg_ = config;
}

}
