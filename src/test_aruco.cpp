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

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

using namespace cv;
using namespace std;
using namespace Eigen;

// #define USE_MY_PNP

Eigen::Matrix3d inner;
Eigen::Vector4d distor;

string datapath = "/home/sam/Downloads/";

int main()
{
  cv::Mat intrinsic_matrix, dist_coef;
  intrinsic_matrix = (cv::Mat_<double>(3, 3) <<
    1564.276649727154, 0, 1017.756959642585,
    0, 1557.311930961359, 777.5116876541903,
    0, 0, 1);
  dist_coef = (cv::Mat_<double>(1, 5) <<
    0.1762349550753628, -0.6758345494648214, 0.002353941090854779, -0.0005721213813597377, 1.05480625138201);
  cv::Mat inputImage = cv::imread(datapath + "IMG_8057.JPEG");

  Size image_size = inputImage.size();
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
  cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
  cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
  cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

  cv::Mat pinhole_distcoe;
  std::vector<cv::Vec3d> rvecs, tvecs;
  Matrix3d rotations;
  Vector3d translations;
  cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.15, intrinsic_matrix, dist_coef, rvecs, tvecs);

  cv::Matx33d rotationMatrix;
  cv::Rodrigues(rvecs[0], rotationMatrix);
  for(int j = 0; j < 3; j++)
  {
    for(int k = 0; k < 3; k++)
      rotations(j, k) = rotationMatrix(j, k);
    translations(j) = tvecs[0](j);
  }
  cout<<rotations<<endl;
  cout<<translations<<endl;
}