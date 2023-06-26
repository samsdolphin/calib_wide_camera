#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>

#include <ros/ros.h>

#include <Eigen/Dense>
#include "ceres/ceres.h"

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

struct Camera
{
  int id;
  Eigen::Matrix3d intrinsic;
  Eigen::Vector4d distortion;
  cv::Mat cv_intrinsic;
  cv::Mat cv_distortion;
};

struct VPnPData
{
  Eigen::Vector3d p;
  double u, v;
};

struct TagData
{
  double x, y, theta;
};

struct MarkerPair
{
  int image_id, ref_tag_id, cam_id1, cam_id2;
  std::vector<int> marker_ids;
  std::vector<std::vector<cv::Point2f>> marker_corners;
  MarkerPair(int img_id_, int tag_id_, std::vector<int> mar_ids_, std::vector<std::vector<cv::Point2f>> corners_,
             int cam_id1_ = -1, int cam_id2_ = -1)
  {
    cam_id1 = cam_id1_;
    cam_id2 = cam_id2_;
    image_id = img_id_;
    ref_tag_id = tag_id_;
    marker_ids = mar_ids_;
    marker_corners = corners_;
  }
};

template<typename Scalar>
Eigen::Matrix<Scalar, 3, 3> R_theta(Scalar theta)
{
  Eigen::Matrix<Scalar, 3, 3> R;
  R << cos(theta), -sin(theta), 0,
       sin(theta), cos(theta), 0,
       0, 0, 1;
  return R;
}