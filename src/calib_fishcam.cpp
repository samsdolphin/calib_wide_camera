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

string datapath = "/media/sam/CR7/20230613_shenzhen_rosbag/calib_fishcam/";

struct MarkerPair
{
  int marker_id;
  Matrix3d tag_R_cam;
  Vector3d tag_t_cam;
  Matrix<double, 6, 1> cov;
  MarkerPair(int id, Matrix3d R_, Vector3d t_, Matrix<double, 6, 1> cov_)
  {
    marker_id = id;
    tag_R_cam = R_;
    tag_t_cam = t_;
    cov = cov_;
  }
};

vector<string> read_filenames(string filepath)
{
  std::vector<std::string> result;
  for(const auto& entry: std::filesystem::directory_iterator(filepath))
    result.push_back(entry.path());
  return result;
}

void obtain_relative_Rt(cv::Mat inputImage, cv::Mat fisheye_cammatix, cv::Mat fisheye_distcoe, vector<MarkerPair>& marker_pairs)
{
  cv::Mat outputImage = inputImage.clone();
  Size image_size = inputImage.size();
  Mat mapx = Mat(image_size, CV_32FC1);
  Mat mapy = Mat(image_size, CV_32FC1);
  Mat R = Mat::eye(3, 3, CV_32F);
  fisheye::initUndistortRectifyMap(fisheye_cammatix, fisheye_distcoe, R, fisheye_cammatix, image_size, CV_32FC1, mapx, mapy);
  cv::remap(outputImage, outputImage, mapx, mapy, INTER_LINEAR);
  // cv::imwrite(datapath + "fr/u3.png", outputImage);

  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
  cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
  cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
  cv::aruco::detectMarkers(outputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

  cv::Mat pinhole_distcoe;
  pinhole_distcoe = (cv::Mat_<double>(1, 4) << 0, 0, 0, 0);
  std::vector<cv::Vec3d> rvecs, tvecs;
  vector<Matrix3d> rotations; rotations.resize(markerIds.size());
  vector<Vector3d> translations; translations.resize(markerIds.size());
  cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.15, fisheye_cammatix, pinhole_distcoe, rvecs, tvecs);

  vector<Quaterniond> init_qs(markerCorners.size());
  vector<Vector3d> init_ts(markerCorners.size());

  for(int i = 0; i < markerIds.size(); i++)
  {
    cv::Matx33d rotationMatrix;
    cv::Rodrigues(rvecs[i], rotationMatrix);

    for(int j = 0; j < 3; j++)
    {
      for(int k = 0; k < 3; k++)
        rotations[i](j, k) = rotationMatrix(j, k);
      translations[i](j) = tvecs[i](j);
    }
    init_qs[i] = Quaterniond(rotations[i]);
    init_ts[i] = Vector3d(translations[i](0), translations[i](1), translations[i](2));
  }

  bool is_first = false;
  for(int i = 1; i < markerCorners.size(); i++)
  {
    if(markerIds[0] > 7 || markerIds[i] > 7) continue;
    
    double delta_theta = init_qs[0].toRotationMatrix().col(2).dot(init_qs[i].toRotationMatrix().col(2));
    double delta_degree = acos(delta_theta)*180/M_PI;
    
    double delta_tz = (init_qs[0].inverse()*(init_ts[i]-init_ts[0]))(2);

    // if(delta_degree > 1.5) continue;
    // if(fabs(delta_tz) > 0.01) continue;

    cout<<"id "<<markerIds[0]<<"-"<<markerIds[i]<<" ";
    cout<<"cos "<<delta_theta<<" | ";
    cout<<"degree "<<delta_degree<<" | ";
    cout<<"delta_tz "<<(init_qs[0].inverse()*(init_ts[i]-init_ts[0]))(2)<<endl;

    Matrix<double, 6, 1> cov;
    cov << fabs(1.0/delta_degree), fabs(1.0/delta_degree), fabs(1.0/delta_degree),
           fabs(1.0/delta_tz), fabs(1.0/delta_tz), fabs(1.0/delta_tz);
    if(!is_first)
    {
      marker_pairs.push_back(MarkerPair(markerIds[0], init_qs[i].toRotationMatrix(), init_ts[i], cov));
      is_first = true;
      cout<<"pushed "<<markerIds[0]<<endl;
    }
    marker_pairs.push_back(MarkerPair(markerIds[i], init_qs[i].toRotationMatrix(), init_ts[i], cov));
    cout<<"pushed "<<markerIds[i]<<endl;
  }
}

int main()
{
  vector<Quaterniond> q_gts(8);
  vector<Vector3d> t_gts(8);

  q_gts[0] = Quaterniond(1, 3.07161e-23, -3.56374e-24, 2.03922e-24);
  t_gts[0] = Vector3d(-1.70008e-28, 9.2677e-28, -1.78625e-27);

  q_gts[1] = Quaterniond(0.999965, 0.00747164, 0.00257681, 0.00291436);
  t_gts[1] = Vector3d(0.482585, 0.00833866, -0.00165951);
  
  q_gts[2] = Quaterniond(0.999997, 0.00208325, 0.00129624, -0.000337695);
  t_gts[2] = Vector3d(0.974004, 0.014019, -0.00149945);
  
  q_gts[3] = Quaterniond(0.999865, -0.0112679, 0.00158151, 0.0118604);
  t_gts[3] = Vector3d(1.43671, 0.027671, 0.00550261);
  
  q_gts[4] = Quaterniond(0.999829, -0.011012, -0.00348125, 0.0144596);
  t_gts[4] = Vector3d(1.91084, 0.0401467, 0.00144359);
  
  q_gts[5] = Quaterniond(0.999956, -0.00859192, -0.00335648, -0.00148587);
  t_gts[5] = Vector3d(2.42921, 0.053376, 0.0018421);
  
  q_gts[6] = Quaterniond(0.999948, -0.00766276, -0.00626059, 0.00243957);
  t_gts[6] = Vector3d(2.92163, 0.0527573, 0.00878065);
  
  q_gts[7] = Quaterniond(0.999901, -0.00193719, -0.0139093, 0.00064138);
  t_gts[7] = Vector3d(3.42418, 0.0326239, -0.00206162);

  cv::Mat fr_intrinsic, fr_coef, bl_intrinsic, bl_coef, fl_intrinsic, fl_coef, br_intrinsic, br_coef;

  fr_intrinsic = (cv::Mat_<double>(3, 3) <<
    1418.361670309597, 0, 1253.973935943561,
    0, 1419.101900308538, 1036.466628861505,
    0, 0, 1);
  fr_coef = (cv::Mat_<double>(1, 4) << -8.45658e-05, -0.00619387, -0.00286654, 0.00127071);

  fl_intrinsic = (cv::Mat_<double>(3, 3) <<
    1420.341348618206, 0, 1224.37438458383,
    0, 1420.997567384703, 1010.762813735306,
    0, 0, 1);
  fl_coef = (cv::Mat_<double>(1, 4) << -0.00425799, 0.00307152, -0.0155525, 0.00805682);

  br_intrinsic = (cv::Mat_<double>(3, 3) <<
    1418.771097125488, 0, 1212.215584588221,
    0, 1419.5068407428, 1042.056348573678,
    0, 0, 1);
  br_coef = (cv::Mat_<double>(1, 4) << -0.00362874, 0.00406696, -0.0204213, 0.0122873);

  bl_intrinsic = (cv::Mat_<double>(3, 3) <<
    1420.153047506825, 0, 1248.175743932881,
    0, 1421.026042801145, 1016.806763168581,
    0, 0, 1);
  bl_coef = (cv::Mat_<double>(1, 4) << 0.000413327, -0.017352, 0.0175895, -0.0110053);

  gtsam::Values initial;
  gtsam::NonlinearFactorGraph graph;
  gtsam::Vector Vector6(6);
  Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8;
  gtsam::noiseModel::Diagonal::shared_ptr priorModel = gtsam::noiseModel::Diagonal::Variances(Vector6);
  initial.insert(0, gtsam::Pose3(gtsam::Rot3(Eigen::MatrixXd::Identity(3, 3)), gtsam::Point3(Eigen::Vector3d::Zero())));
  graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(gtsam::Rot3(Eigen::MatrixXd::Identity(3, 3)),
                                                             gtsam::Point3(Eigen::Vector3d::Zero())), priorModel));
  Matrix3d R; R << 0, 0, -1, 0, 1, 0, 1, 0, 0;
  initial.insert(1, gtsam::Pose3(gtsam::Rot3(R), gtsam::Point3(Eigen::Vector3d::Zero())));
  initial.insert(2, gtsam::Pose3(gtsam::Rot3(R*R), gtsam::Point3(Eigen::Vector3d::Zero())));
  initial.insert(3, gtsam::Pose3(gtsam::Rot3(R.transpose()), gtsam::Point3(Eigen::Vector3d::Zero())));
  // cout<<"_______________________________"<<endl;
  // Quaterniond qtmp(R); 
  // cout<<qtmp.x()<<" "<<qtmp.y()<<" "<<qtmp.z()<<" "<<qtmp.w()<<endl;
  // qtmp = Quaterniond(R*R);
  // cout<<qtmp.x()<<" "<<qtmp.y()<<" "<<qtmp.z()<<" "<<qtmp.w()<<endl;
  // qtmp = Quaterniond(R.transpose());
  // cout<<qtmp.x()<<" "<<qtmp.y()<<" "<<qtmp.z()<<" "<<qtmp.w()<<endl;
  // cout<<"_______________________________"<<endl;

  cout<<"BL-FL"<<endl;
  vector<string> bl_img_names = read_filenames(datapath + "bl_fl/bl/");
  vector<string> fl_img_names = read_filenames(datapath + "bl_fl/fl/");
  int total_cnt = 0, valid_cnt = 0;
  for(int k = 0; k < bl_img_names.size(); k++)
  {
    vector<MarkerPair> bl_marker_pairs, fl_marker_pairs;
    cv::Mat inputImage = cv::imread(bl_img_names[k]);
    obtain_relative_Rt(inputImage, bl_intrinsic, bl_coef, bl_marker_pairs);
    inputImage = cv::imread(fl_img_names[k]);
    obtain_relative_Rt(inputImage, fl_intrinsic, fl_coef, fl_marker_pairs);
    
    if(bl_marker_pairs.size() > 0 && fl_marker_pairs.size() > 0)
      for(int i = 0; i < bl_marker_pairs.size(); i++)
        for(int j = 0; j < fl_marker_pairs.size(); j++)
        {
          total_cnt++;
          int tag_id1 = bl_marker_pairs[i].marker_id;
          int tag_id2 = fl_marker_pairs[j].marker_id;
          Quaterniond qab = q_gts[tag_id1].inverse() * q_gts[tag_id2];
          Vector3d tab = q_gts[tag_id1].inverse() * (t_gts[tag_id2] - t_gts[tag_id1]);
          Matrix3d delta_R = bl_marker_pairs[i].tag_R_cam * qab * fl_marker_pairs[j].tag_R_cam.transpose();
          Vector3d delta_t = bl_marker_pairs[i].tag_R_cam * tab + bl_marker_pairs[i].tag_t_cam - delta_R * fl_marker_pairs[j].tag_t_cam;
          
          if(delta_t(0) < 0 && delta_t(2) < 0 && fabs(delta_t(1)) < 0.03)
          {
            // cout<<"BL-FL "<<tag_id1<<"-"<<tag_id2<<endl;
            // cout<<delta_R<<endl<<endl;
            // cout<<delta_t.transpose()<<endl<<endl;
            gtsam::Rot3 R_sam(delta_R);
            gtsam::Point3 t_sam(delta_t);
            Vector6 << (bl_marker_pairs[i].cov(0) + fl_marker_pairs[j].cov(0))/2, (bl_marker_pairs[i].cov(1) + fl_marker_pairs[j].cov(1))/2,
                       (bl_marker_pairs[i].cov(2) + fl_marker_pairs[j].cov(2))/2, (bl_marker_pairs[i].cov(3) + fl_marker_pairs[j].cov(3))/2,
                       (bl_marker_pairs[i].cov(4) + fl_marker_pairs[j].cov(4))/2, (bl_marker_pairs[i].cov(5) + fl_marker_pairs[j].cov(5))/2;
            gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
            gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(1, 2,
                                                                                             gtsam::Pose3(R_sam, t_sam), odometryNoise));
            graph.push_back(factor);
            valid_cnt++;
          }
        }
  }
  cout<<"valid "<<valid_cnt<<"/"<<total_cnt<<endl<<endl;

  cout<<"BR-FR"<<endl;
  vector<string> br_img_names = read_filenames(datapath + "br_fr/br/");
  vector<string> fr_img_names = read_filenames(datapath + "br_fr/fr/");
  total_cnt = 0, valid_cnt = 0;
  for(int k = 0; k < br_img_names.size(); k++)
  {
    vector<MarkerPair> br_marker_pairs, fr_marker_pairs;
    cv::Mat inputImage = cv::imread(br_img_names[k]);
    obtain_relative_Rt(inputImage, br_intrinsic, br_coef, br_marker_pairs);
    inputImage = cv::imread(fr_img_names[k]);
    obtain_relative_Rt(inputImage, fr_intrinsic, fr_coef, fr_marker_pairs);

    if(br_marker_pairs.size() > 0 && fr_marker_pairs.size() > 0)
      for(int i = 0; i < br_marker_pairs.size(); i++)
        for(int j = 0; j < fr_marker_pairs.size(); j++)
        {
          total_cnt++;
          int tag_id1 = br_marker_pairs[i].marker_id;
          int tag_id2 = fr_marker_pairs[j].marker_id;
          Quaterniond qab = q_gts[tag_id1].inverse() * q_gts[tag_id2];
          Vector3d tab = q_gts[tag_id1].inverse() * (t_gts[tag_id2] - t_gts[tag_id1]);
          Matrix3d delta_R = br_marker_pairs[i].tag_R_cam * qab * fr_marker_pairs[j].tag_R_cam.transpose();
          Vector3d delta_t = br_marker_pairs[i].tag_R_cam * tab + br_marker_pairs[i].tag_t_cam - delta_R * fr_marker_pairs[j].tag_t_cam;

          if(delta_t(0) < 0 && delta_t(2) < 0 && fabs(delta_t(1)) < 0.03)
          {
            gtsam::Rot3 R_sam(delta_R);
            gtsam::Point3 t_sam(delta_t);
            Vector6 << (br_marker_pairs[i].cov(0) + fr_marker_pairs[j].cov(0))/2, (br_marker_pairs[i].cov(1) + fr_marker_pairs[j].cov(1))/2,
                       (br_marker_pairs[i].cov(2) + fr_marker_pairs[j].cov(2))/2, (br_marker_pairs[i].cov(3) + fr_marker_pairs[j].cov(3))/2,
                       (br_marker_pairs[i].cov(4) + fr_marker_pairs[j].cov(4))/2, (br_marker_pairs[i].cov(5) + fr_marker_pairs[j].cov(5))/2;
            gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
            gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(3, 0,
                                                                                             gtsam::Pose3(R_sam, t_sam), odometryNoise));
            graph.push_back(factor);
            valid_cnt++;
          }
        }
  }
  cout<<"valid "<<valid_cnt<<"/"<<total_cnt<<endl<<endl;

  cout<<"FL-BR"<<endl;
  fl_img_names = read_filenames(datapath + "fl_br/fl/");
  br_img_names = read_filenames(datapath + "fl_br/br/");
  total_cnt = 0, valid_cnt = 0;
  for(int k = 0; k < fl_img_names.size(); k++)
  {
    vector<MarkerPair> fl_marker_pairs, br_marker_pairs;
    cv::Mat inputImage = cv::imread(fl_img_names[k]);
    obtain_relative_Rt(inputImage, fl_intrinsic, fl_coef, fl_marker_pairs);
    inputImage = cv::imread(br_img_names[k]);
    obtain_relative_Rt(inputImage, br_intrinsic, br_coef, br_marker_pairs);

    if(fl_marker_pairs.size() > 0 && br_marker_pairs.size() > 0)
      for(int i = 0; i < fl_marker_pairs.size(); i++)
        for(int j = 0; j < br_marker_pairs.size(); j++)
        {
          total_cnt++;
          int tag_id1 = fl_marker_pairs[i].marker_id;
          int tag_id2 = br_marker_pairs[j].marker_id;
          Quaterniond qab = q_gts[tag_id1].inverse() * q_gts[tag_id2];
          Vector3d tab = q_gts[tag_id1].inverse() * (t_gts[tag_id2] - t_gts[tag_id1]);
          Matrix3d delta_R = fl_marker_pairs[i].tag_R_cam * qab * br_marker_pairs[j].tag_R_cam.transpose();
          Vector3d delta_t = fl_marker_pairs[i].tag_R_cam * tab + fl_marker_pairs[i].tag_t_cam - delta_R * br_marker_pairs[j].tag_t_cam;

          if(delta_t(0) < 0 && delta_t(2) < 0 && fabs(delta_t(1)) < 0.03)
          {
            gtsam::Rot3 R_sam(delta_R);
            gtsam::Point3 t_sam(delta_t);
            Vector6 << (fl_marker_pairs[i].cov(0) + br_marker_pairs[j].cov(0))/2, (fl_marker_pairs[i].cov(1) + br_marker_pairs[j].cov(1))/2,
                       (fl_marker_pairs[i].cov(2) + br_marker_pairs[j].cov(2))/2, (fl_marker_pairs[i].cov(3) + br_marker_pairs[j].cov(3))/2,
                       (fl_marker_pairs[i].cov(4) + br_marker_pairs[j].cov(4))/2, (fl_marker_pairs[i].cov(5) + br_marker_pairs[j].cov(5))/2;
            gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
            gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(2, 3,
                                                                                             gtsam::Pose3(R_sam, t_sam), odometryNoise));
            graph.push_back(factor);
            valid_cnt++;
          }
        }
  }
  cout<<"valid "<<valid_cnt<<"/"<<total_cnt<<endl<<endl;

  cout<<"FR-BL"<<endl;
  fr_img_names = read_filenames(datapath + "fr_bl/fr/");
  bl_img_names = read_filenames(datapath + "fr_bl/bl/");
  total_cnt = 0, valid_cnt = 0;
  for(int k = 0; k < fr_img_names.size(); k++)
  {
    vector<MarkerPair> fr_marker_pairs, bl_marker_pairs;
    cv::Mat inputImage = cv::imread(fr_img_names[k]);
    obtain_relative_Rt(inputImage, fr_intrinsic, fr_coef, fr_marker_pairs);
    inputImage = cv::imread(bl_img_names[k]);
    obtain_relative_Rt(inputImage, bl_intrinsic, bl_coef, bl_marker_pairs);
    
    if(fr_marker_pairs.size() > 0 && bl_marker_pairs.size() > 0)
      for(int i = 0; i < fr_marker_pairs.size(); i++)
        for(int j = 0; j < bl_marker_pairs.size(); j++)
        {
          total_cnt++;
          int tag_id1 = fr_marker_pairs[i].marker_id;
          int tag_id2 = bl_marker_pairs[j].marker_id;
          Quaterniond qab = q_gts[tag_id1].inverse() * q_gts[tag_id2];
          Vector3d tab = q_gts[tag_id1].inverse() * (t_gts[tag_id2] - t_gts[tag_id1]);
          Matrix3d delta_R = fr_marker_pairs[i].tag_R_cam * qab * bl_marker_pairs[j].tag_R_cam.transpose();
          Vector3d delta_t = fr_marker_pairs[i].tag_R_cam * tab + fr_marker_pairs[i].tag_t_cam - delta_R * bl_marker_pairs[j].tag_t_cam;
          
          if(delta_t(0) < 0 && delta_t(2) < 0 && fabs(delta_t(1)) < 0.03)
          {
            // cout<<"BL-FL "<<tag_id1<<"-"<<tag_id2<<endl;
            // cout<<delta_R<<endl<<endl;
            // cout<<delta_t.transpose()<<endl<<endl;
            gtsam::Rot3 R_sam(delta_R);
            gtsam::Point3 t_sam(delta_t);
            Vector6 << (fr_marker_pairs[i].cov(0) + bl_marker_pairs[j].cov(0))/2, (fr_marker_pairs[i].cov(1) + bl_marker_pairs[j].cov(1))/2,
                       (fr_marker_pairs[i].cov(2) + bl_marker_pairs[j].cov(2))/2, (fr_marker_pairs[i].cov(3) + bl_marker_pairs[j].cov(3))/2,
                       (fr_marker_pairs[i].cov(4) + bl_marker_pairs[j].cov(4))/2, (fr_marker_pairs[i].cov(5) + bl_marker_pairs[j].cov(5))/2;
            gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
            gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(0, 1,
                                                                                             gtsam::Pose3(R_sam, t_sam), odometryNoise));
            graph.push_back(factor);
            valid_cnt++;
          }
        }
  }
  cout<<"valid "<<valid_cnt<<"/"<<total_cnt<<endl<<endl;

  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  gtsam::ISAM2 isam(parameters);
  isam.update(graph, initial);
  isam.update();

  gtsam::Values results = isam.calculateEstimate();
  
  for(uint i = 0; i < results.size(); i++)
  {
    gtsam::Pose3 pose = results.at(i).cast<gtsam::Pose3>();
    cout<<"cam "<<i<<endl;
    Quaterniond q(pose.rotation().matrix());
    Vector3d t = pose.translation();
    // cout<<q.w()<<", "<<q.x()<<", "<<q.y()<<", "<<q.z()<<", "<<t(0)<<", "<<t(1)<<", "<<t(2)<<endl;
    cout<<"R "<<pose.rotation().matrix()<<endl;
    cout<<"t "<<pose.translation()<<endl<<endl;
  }

  Matrix3d R_fr_L;
  R_fr_L << -1, 0, 0, 0, 0, 1, 0, 1, 0;
  Vector3d t_fr_L(0.0695, 0.06885, -0.0765);

  Matrix3d R_br_L;
  R_br_L << 0, -1, 0, 0, 0, 1, -1, 0, 0;
  Vector3d t_br_L(0.0165, 0.06885, -0.1256);

  Matrix3d R_bl_L;
  R_bl_L << 0, 1, 0, 0, 0, 1, 1, 0, 0;
  Vector3d t_bl_L(0.045, 0.06885, -0.1256);

  Matrix3d R_fl_L;
  R_fl_L << 1, 0, 0, 0, 0, 1, 0, -1, 0;
  Vector3d t_fl_L(0, 0.06885, -0.1256);

  Matrix3d delta_R = R_fr_L * R_bl_L.transpose();
  Vector3d delta_t = t_fr_L - delta_R * t_bl_L;
  cout<<"BL"<<endl;
  cout<<delta_R<<endl;
  cout<<delta_t.transpose()<<endl<<endl;

  delta_R = R_fr_L * R_fl_L.transpose();
  delta_t = t_fr_L - delta_R * t_fl_L;
  cout<<"FL"<<endl;
  cout<<delta_R<<endl;
  cout<<delta_t.transpose()<<endl<<endl;

  delta_R = R_fr_L * R_br_L.transpose();
  delta_t = t_fr_L - delta_R * t_br_L;
  cout<<"BR"<<endl;
  cout<<delta_R<<endl;
  cout<<delta_t.transpose()<<endl<<endl;
}