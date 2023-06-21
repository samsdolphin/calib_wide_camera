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

#define USE_MY_PNP

Eigen::Matrix3d inner;
Eigen::Vector4d distor;

int img_num = 0;

string datapath = "/media/sam/CR7/20230613_shenzhen_rosbag/calib_aruco/";

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
  int id1, id2;
  Eigen::Matrix3d delta_R;
  Eigen::Vector3d delta_t;
  Eigen::Matrix<double, 6, 1> cov;
  MarkerPair(int id1_, int id2_, Matrix3d R_, Vector3d t_, Matrix<double, 6, 1> cov_)
  {
    id1 = id1_;
    id2 = id2_;
    delta_R = R_;
    delta_t = t_;
    cov = cov_;
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

class two_tag_extrinsic
{
public:
  two_tag_extrinsic(VPnPData p, TagData ti, TagData tj, Quaterniond q, Vector3d t) {pd = p; tdi = ti; tdj = tj; _q = q; _t = t;}

  template <typename T>
  bool operator()(const T* _tag1, const T* _tag2, T* residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
    Eigen::Quaternion<T> q_incre = _q.cast<T>();
    Eigen::Matrix<T, 3, 1> t_incre = _t.cast<T>();
    Eigen::Matrix<T, 3, 1> p_l(T(pd.p(0)), T(pd.p(1)), T(pd.p(2)));
    
    Eigen::Matrix<T, 3, 3> delta_R = R_theta(_tag1[2]).transpose()*R_theta(_tag2[2]);
    Eigen::Matrix<T, 3, 1> delta_t = R_theta(_tag1[2]).transpose()*(Eigen::Matrix<T, 3, 1>(_tag2[0]-_tag1[0], _tag2[1]-_tag1[1], T(0)));
    p_l = delta_R * p_l + delta_t;

    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
    const T& fx = innerT.coeffRef(0, 0);
    const T& cx = innerT.coeffRef(0, 2);
    const T& fy = innerT.coeffRef(1, 1);
    const T& cy = innerT.coeffRef(1, 2);
    T a = p_c[0] / p_c[2];
    T b = p_c[1] / p_c[2];
    T r = sqrt(a * a + b * b);
    T theta = atan(r);
    T theta_d = theta * (T(1) + distorT[0] * pow(theta, T(2)) + distorT[1] * pow(theta, T(4)) +
      distorT[2] * pow(theta, T(6)) + distorT[3] * pow(theta, T(8)));

    T dx = (theta_d / r) * a;
    T dy = (theta_d / r) * b;
    T ud = fx * dx + cx;
    T vd = fy * dy + cy;
    residuals[0] = ud - T(pd.u);
    residuals[1] = vd - T(pd.v);

    return true;
  }
  
  static ceres::CostFunction *Create(VPnPData p, TagData ti, TagData tj, Quaterniond q, Vector3d t)
  {
    return (new ceres::AutoDiffCostFunction<two_tag_extrinsic, 2, 3, 3>(new two_tag_extrinsic(p, ti, tj, q, t)));
  }

private:
  VPnPData pd;
  TagData tdi, tdj;
  Quaterniond _q;
  Vector3d _t;
};

class two_tag_residual
{
public:
  two_tag_residual(VPnPData p, TagData ti, TagData tj) {pd = p; tdi = ti; tdj = tj;}

  template <typename T>
  bool operator()(const T* _q, const T* _t, const T* _tag1, const T* _tag2, T* residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
    Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
    Eigen::Matrix<T, 3, 1> p_l(T(pd.p(0)), T(pd.p(1)), T(pd.p(2)));
    
    Eigen::Matrix<T, 3, 3> delta_R = R_theta(_tag1[2]).transpose()*R_theta(_tag2[2]);
    Eigen::Matrix<T, 3, 1> delta_t = R_theta(_tag1[2]).transpose()*(Eigen::Matrix<T, 3, 1>(_tag2[0]-_tag1[0], _tag2[1]-_tag1[1], T(0)));
    p_l = delta_R * p_l + delta_t;

    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
    const T& fx = innerT.coeffRef(0, 0);
    const T& cx = innerT.coeffRef(0, 2);
    const T& fy = innerT.coeffRef(1, 1);
    const T& cy = innerT.coeffRef(1, 2);
    T a = p_c[0] / p_c[2];
    T b = p_c[1] / p_c[2];
    T r = sqrt(a * a + b * b);
    T theta = atan(r);
    T theta_d = theta * (T(1) + distorT[0] * pow(theta, T(2)) + distorT[1] * pow(theta, T(4)) +
      distorT[2] * pow(theta, T(6)) + distorT[3] * pow(theta, T(8)));

    T dx = (theta_d / r) * a;
    T dy = (theta_d / r) * b;
    T ud = fx * dx + cx;
    T vd = fy * dy + cy;
    residuals[0] = ud - T(pd.u);
    residuals[1] = vd - T(pd.v);

    return true;
  }
  
  static ceres::CostFunction *Create(VPnPData p, TagData ti, TagData tj)
  {
    return (new ceres::AutoDiffCostFunction<two_tag_residual, 2, 4, 3, 3, 3>(new two_tag_residual(p, ti, tj)));
  }

private:
  VPnPData pd;
  TagData tdi, tdj;
};

class one_tag_residual
{
public:
  one_tag_residual(VPnPData p) {pd = p;}

  template <typename T>
  bool operator()(const T* _q, const T* _t, T* residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
    Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
    Eigen::Matrix<T, 3, 1> p_l(T(pd.p(0)), T(pd.p(1)), T(pd.p(2)));
    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
    const T& fx = innerT.coeffRef(0, 0);
    const T& cx = innerT.coeffRef(0, 2);
    const T& fy = innerT.coeffRef(1, 1);
    const T& cy = innerT.coeffRef(1, 2);
    T a = p_c[0] / p_c[2];
    T b = p_c[1] / p_c[2];
    T r = sqrt(a * a + b * b);
    T theta = atan(r);
    T theta_d = theta * (T(1) + distorT[0] * pow(theta, T(2)) + distorT[1] * pow(theta, T(4)) +
      distorT[2] * pow(theta, T(6)) + distorT[3] * pow(theta, T(8)));

    T dx = (theta_d / r) * a;
    T dy = (theta_d / r) * b;
    T ud = fx * dx + cx;
    T vd = fy * dy + cy;
    residuals[0] = ud - T(pd.u);
    residuals[1] = vd - T(pd.v);

    return true;
  }
  
  static ceres::CostFunction *Create(VPnPData p)
  {
    return (new ceres::AutoDiffCostFunction<one_tag_residual, 2, 4, 3>(new one_tag_residual(p)));
  }

private:
  VPnPData pd;
};

/*每个tag和tag0的外参*/
double tag_ext[24];

void obtain_relative_Rt(cv::Mat inputImage, cv::Mat fisheye_cammatix, cv::Mat fisheye_distcoe, vector<MarkerPair>& marker_pairs)
{
  cv::Mat outputImage = inputImage.clone();
  #ifndef USE_MY_PNP
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

  // if(markerIds.size() > 0)
  //   cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);
  // cv::imwrite(datapath + to_string(img_num++) +".png", outputImage);

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

  for(int i = 1; i < markerCorners.size(); i++)
  {
    if(markerIds[0] > 7 || markerIds[i] > 7) continue;
    
    double delta_theta = init_qs[0].toRotationMatrix().col(2).dot(init_qs[i].toRotationMatrix().col(2));
    double delta_degree = acos(delta_theta)*180/M_PI;
    
    double delta_tz = (init_qs[0].inverse()*(init_ts[i]-init_ts[0]))(2);

    // if(delta_degree > 1.5) continue;
    if(fabs(delta_tz) > 0.01) continue;

    cout<<"id "<<markerIds[0]<<"-"<<markerIds[i]<<" ";
    cout<<"cos "<<delta_theta<<" | ";
    cout<<"degree "<<delta_degree<<" | ";
    cout<<"delta_tz "<<(init_qs[0].inverse()*(init_ts[i]-init_ts[0]))(2)<<endl;

    Matrix<double, 6, 1> cov;
    cov << fabs(1.0/delta_degree), fabs(1.0/delta_degree), fabs(1.0/delta_degree),
           fabs(1.0/delta_tz), fabs(1.0/delta_tz), fabs(1.0/delta_tz);
    marker_pairs.push_back(MarkerPair(markerIds[0], markerIds[i],
                                      rotations[0].transpose()*rotations[i], rotations[0].transpose()*(translations[i]-translations[0]),
                                      cov));
  }
  #else
  /* Rought Estimation */
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
  cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
  cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
  cv::aruco::detectMarkers(outputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

  // if(markerIds.size() > 0)
  //   cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);
  // cv::imwrite(datapath + to_string(img_num++) +".png", outputImage);

  std::vector<cv::Vec3d> rvecs, tvecs;
  vector<Matrix3d> rotations; rotations.resize(markerIds.size());
  vector<Vector3d> translations; translations.resize(markerIds.size());
  cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.15, fisheye_cammatix, fisheye_distcoe, rvecs, tvecs);

  vector<Quaterniond> init_qs(markerCorners.size()), final_qs(markerCorners.size());
  vector<Vector3d> init_ts(markerCorners.size()), final_ts(markerCorners.size());
  
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

  /* Refine Optimization */
  inner << 1418.361670309597, 0, 1253.973935943561,
           0, 1419.101900308538, 1036.466628861505,
           0, 0, 1;
  distor << -8.45658e-05, -0.00619387, -0.00286654, 0.00127071; // FR

  double res = 0;
  for(int i = 0; i < markerCorners.size(); i++)
  {
    double ext[7] = {init_qs[i].x(), init_qs[i].y(), init_qs[i].z(), init_qs[i].w(), init_ts[i](0), init_ts[i](1), init_ts[i](2)}; // qx, qy, qz, qw, tx, ty, tz

    vector<VPnPData> vpnp_data(4);
    vpnp_data[0].p = Eigen::Vector3d(0, 0, 0); vpnp_data[0].u = markerCorners[i][0].x; vpnp_data[0].v = markerCorners[i][0].y;
    vpnp_data[1].p = Eigen::Vector3d(0.15, 0, 0); vpnp_data[1].u = markerCorners[i][1].x; vpnp_data[1].v = markerCorners[i][1].y;
    vpnp_data[2].p = Eigen::Vector3d(0.15, -0.15, 0); vpnp_data[2].u = markerCorners[i][2].x; vpnp_data[2].v = markerCorners[i][2].y;
    vpnp_data[3].p = Eigen::Vector3d(0, -0.15, 0); vpnp_data[3].u = markerCorners[i][3].x; vpnp_data[3].v = markerCorners[i][3].y;

    ceres::Problem problem;
    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
    problem.AddParameterBlock(ext, 4, q_parameterization);
    problem.AddParameterBlock(ext + 4, 3);
    
    Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(ext);
    Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(ext + 4);

    for(int j = 0; j < 4; j++)
    {
      ceres::CostFunction *cost_function;
      cost_function = one_tag_residual::Create(vpnp_data[j]);
      problem.AddResidualBlock(cost_function, NULL, ext, ext + 4);
    }

    ceres::Solver::Options options;
    options.preconditioner_type = ceres::JACOBI;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.BriefReport() << std::endl;

    final_qs[i] = m_q;
    final_ts[i] = m_t;

    vector<cv::Point3f> pts_3d;
    pts_3d.push_back(cv::Point3f(0, 0, 0));
    pts_3d.push_back(cv::Point3f(0.15, 0, 0));
    pts_3d.push_back(cv::Point3f(0.15, -0.15, 0));
    pts_3d.push_back(cv::Point3f(0, -0.15, 0));
    std::vector<cv::Point2f> pts_2d;
    Eigen::Matrix3d rotation = m_q.toRotationMatrix();
    cv::Mat R = (cv::Mat_<double>(3,3) <<
      rotation(0, 0), rotation(0, 1), rotation(0, 2),
      rotation(1, 0), rotation(1, 1), rotation(1, 2),
      rotation(2, 0), rotation(2, 1), rotation(2, 2));
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    cv::Mat tvec = (cv::Mat_<double>(3, 1) << m_t(0), m_t(1), m_t(2));
    cv::fisheye::projectPoints(pts_3d, pts_2d, rvec, tvec, fisheye_cammatix, fisheye_distcoe);
    for(int j = 0; j < pts_2d.size(); j++)
    {
      // cv::circle(outputImage, pts_2d[j], 2, cv::Scalar(0, 0, 255), -1);
      res += cv::norm(markerCorners[i][j] - pts_2d[j]);
    }
  }
  res /= markerCorners.size()*4;
  cout<<"average reprojection error (pixel): "<<res<<endl;
  // cv::imwrite(datapath + to_string(img_num++) + "_" + to_string(res) + ".png", outputImage);

  for(int i = 0; i < 8; i++)
  {
    Eigen::Matrix3d R0(final_qs[0]);
    Eigen::Vector3d t0 = final_ts[0];
    Eigen::Vector3d t_ = R0.transpose()*(final_ts[i]-t0);
    Eigen::Matrix3d R_ = R0.transpose()*final_qs[i].toRotationMatrix();
    tag_ext[markerIds[i]*3] = t_(0);
    tag_ext[markerIds[i]*3+1] = t_(1);
    tag_ext[markerIds[i]*3+2] = atan2(R_(1, 0), R_(0, 0));
  }
  // for(int i = 0; i < 8; i++)
  // {
  //   cout<<"id "<<i<<endl;
  //   cout<<tag_ext[i*3]<<" "<<tag_ext[i*3+1]<<" "<<tag_ext[i*3+2]<<endl;
  // }

  /*这张图片和相机外参，参考tag是识别的第一个tag*/
  double img_ext[7] = {final_qs[0].x(), final_qs[0].y(), final_qs[0].z(), final_qs[0].w(), final_ts[0](0), final_ts[0](1), final_ts[0](2)};

  ceres::Problem problem;
  // ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
  for(int i = 0; i < 8; i++)
    problem.AddParameterBlock(tag_ext + 3*i, 3);

  int ref_id = markerIds[0]; // 参考tag的id

  for(int i = 0; i < markerCorners.size(); i++)
  {
    vector<VPnPData> vpnp_data(4);
    vpnp_data[0].p = Eigen::Vector3d(0, 0, 0); vpnp_data[0].u = markerCorners[i][0].x; vpnp_data[0].v = markerCorners[i][0].y;
    vpnp_data[1].p = Eigen::Vector3d(0.15, 0, 0); vpnp_data[1].u = markerCorners[i][1].x; vpnp_data[1].v = markerCorners[i][1].y;
    vpnp_data[2].p = Eigen::Vector3d(0.15, -0.15, 0); vpnp_data[2].u = markerCorners[i][2].x; vpnp_data[2].v = markerCorners[i][2].y;
    vpnp_data[3].p = Eigen::Vector3d(0, -0.15, 0); vpnp_data[3].u = markerCorners[i][3].x; vpnp_data[3].v = markerCorners[i][3].y;

    if(i == 0)
      for(int j = 0; j < 4; j++)
      {
        // ceres::CostFunction *cost_function;
        // cost_function = one_tag_residual::Create(vpnp_data[j]);
        // problem.AddResidualBlock(cost_function, NULL, img_ext, img_ext + 4);
      }
    else
    {
      int cur_id = markerIds[i];
      for(int j = 0; j < 4; j++)
      {
        ceres::CostFunction *cost_function;
        cost_function = two_tag_extrinsic::Create(vpnp_data[j],
                                                  TagData{tag_ext[ref_id*3], tag_ext[ref_id*3+1], tag_ext[ref_id*3+2]},
                                                  TagData{tag_ext[cur_id*3], tag_ext[cur_id*3+1], tag_ext[cur_id*3+2]},
                                                  final_qs[0], final_ts[0]);
        problem.AddResidualBlock(cost_function, NULL, tag_ext + 3*ref_id, tag_ext + 3*cur_id);
      }
    }
  }

  ceres::Solver::Options options;
  options.preconditioner_type = ceres::JACOBI;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = 0;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Eigen::Matrix3d R0 = R_theta(tag_ext[ref_id*3+2]);
  Eigen::Vector3d t0(tag_ext[ref_id*3], tag_ext[ref_id*3+1], 0);
  // for(int i = 0; i < 8; i++)
  // {
  //   cout<<"id "<<i<<endl;
  //   cout<<R0.transpose()*R_theta(tag_ext[i*3+2])<<endl;
  //   cout<<R0.transpose()*(Eigen::Vector3d(tag_ext[i*3], tag_ext[i*3+1], 0)-t0)<<endl;
  // }

  Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(img_ext);
  Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(img_ext + 4);

  res = 0;
  for(int i = 0; i < 8; i++)
  {
    int cur_id = markerIds[i];
    Eigen::Matrix3d new_R = R0.transpose()*R_theta(tag_ext[cur_id*3+2]);
    Eigen::Vector3d new_t = R0.transpose()*(Eigen::Vector3d(tag_ext[cur_id*3], tag_ext[cur_id*3+1], 0)-t0);
    // Eigen::Matrix3d new_R = R_theta(tag_ext[cur_id*3+2]);
    // Eigen::Vector3d new_t = Eigen::Vector3d(tag_ext[cur_id*3], tag_ext[cur_id*3+1], 0);

    vector<cv::Point3f> pts_3d;
    Eigen::Vector3d t_tmp;
    t_tmp = new_R * Eigen::Vector3d(0, 0, 0) + new_t;
    pts_3d.push_back(cv::Point3f(t_tmp(0), t_tmp(1), t_tmp(2)));
    t_tmp = new_R * Eigen::Vector3d(0.15, 0, 0) + new_t;
    pts_3d.push_back(cv::Point3f(t_tmp(0), t_tmp(1), t_tmp(2)));
    t_tmp = new_R * Eigen::Vector3d(0.15, -0.15, 0) + new_t;
    pts_3d.push_back(cv::Point3f(t_tmp(0), t_tmp(1), t_tmp(2)));
    t_tmp = new_R * Eigen::Vector3d(0, -0.15, 0) + new_t;
    pts_3d.push_back(cv::Point3f(t_tmp(0), t_tmp(1), t_tmp(2)));
    
    std::vector<cv::Point2f> pts_2d;
    Eigen::Matrix3d rotation = m_q.toRotationMatrix();
    cv::Mat R = (cv::Mat_<double>(3,3) <<
      rotation(0, 0), rotation(0, 1), rotation(0, 2),
      rotation(1, 0), rotation(1, 1), rotation(1, 2),
      rotation(2, 0), rotation(2, 1), rotation(2, 2));
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    cv::Mat tvec = (cv::Mat_<double>(3, 1) << m_t(0), m_t(1), m_t(2));
    cv::fisheye::projectPoints(pts_3d, pts_2d, rvec, tvec, fisheye_cammatix, fisheye_distcoe);
    for(int j = 0; j < pts_2d.size(); j++)
    {
      cv::circle(outputImage, pts_2d[j], 2, cv::Scalar(0, 0, 255), -1);
      res += cv::norm(markerCorners[i][j] - pts_2d[j]);
    }
  }
  res /= markerCorners.size()*4;
  cout<<"average reprojection error (pixel): "<<res<<endl;
  cv::imwrite(datapath + "test.png", outputImage);
  
  

  // for(int i = 1; i < markerCorners.size(); i++)
  // {
  //   cout<<"id "<<markerIds[0]<<"-"<<markerIds[i]<<" ";
  //   cout<<"cos "<<final_qs[0].toRotationMatrix().col(2).dot(final_qs[i].toRotationMatrix().col(2))<<" | ";
  //   cout<<"degree "<<acos(final_qs[0].toRotationMatrix().col(2).dot(final_qs[i].toRotationMatrix().col(2)))*180/M_PI<<" | ";
  //   cout<<"delta_tz "<<(final_qs[0].inverse()*(final_ts[i]-final_ts[0]))(2)<<endl;
  // }
  #endif
}

int main()
{
  vector<MarkerPair> marker_pairs;

  // for(int i = 0; i < 8; i++)
  // {
  //   tag_ext[i*3] = 0.4*i; // delta_x
  //   tag_ext[i*3+1] = 0; // delta_y
  //   tag_ext[i*3+2] = 0; // delta_theta
  // }

  cout<<"FR"<<endl;
  // for(const auto& entry: filesystem::directory_iterator(datapath + "fr/"))
  {
    cv::Mat fisheye_cammatix, fisheye_distcoe;
    fisheye_cammatix = (cv::Mat_<double>(3, 3) <<
      1418.361670309597, 0, 1253.973935943561,
      0, 1419.101900308538, 1036.466628861505,
      0, 0, 1);
    fisheye_distcoe = (cv::Mat_<double>(1, 4) << -8.45658e-05, -0.00619387, -0.00286654, 0.00127071);
    // cv::Mat inputImage = cv::imread(entry.path());
    cv::Mat inputImage = cv::imread(datapath + "fr/3.png");
    obtain_relative_Rt(inputImage, fisheye_cammatix, fisheye_distcoe, marker_pairs);
  }
  exit(0);

  // cout<<"FL"<<endl;
  // for(const auto & entry : filesystem::directory_iterator(datapath + "fl/"))
  // {
  //   cv::Mat fisheye_cammatix, fisheye_distcoe;
  //   fisheye_cammatix = (cv::Mat_<double>(3, 3) <<
  //     1420.341348618206, 0, 1224.37438458383,
  //     0, 1420.997567384703, 1010.762813735306,
  //     0, 0, 1);
  //   fisheye_distcoe = (cv::Mat_<double>(1, 4) << -0.00425799, 0.00307152, -0.0155525, 0.00805682);
  //   cv::Mat inputImage = cv::imread(entry.path());
  //   obtain_relative_Rt(inputImage, fisheye_cammatix, fisheye_distcoe, marker_pairs);
  // }

  // cout<<"BR"<<endl;
  // for(const auto & entry : filesystem::directory_iterator(datapath + "br/"))
  // {
  //   cv::Mat fisheye_cammatix, fisheye_distcoe;
  //   fisheye_cammatix = (cv::Mat_<double>(3, 3) <<
  //     1418.771097125488, 0, 1212.215584588221,
  //     0, 1419.5068407428, 1042.056348573678,
  //     0, 0, 1);
  //   fisheye_distcoe = (cv::Mat_<double>(1, 4) << -0.00362874, 0.00406696, -0.0204213, 0.0122873);
  //   cv::Mat inputImage = cv::imread(entry.path());
  //   obtain_relative_Rt(inputImage, fisheye_cammatix, fisheye_distcoe, marker_pairs);
  // }

  // cout<<"BL"<<endl;
  // for(const auto & entry : filesystem::directory_iterator(datapath + "bl/"))
  // {
  //   cv::Mat fisheye_cammatix, fisheye_distcoe;
  //   fisheye_cammatix = (cv::Mat_<double>(3, 3) <<
  //     1420.153047506825, 0, 1248.175743932881,
  //     0, 1421.026042801145, 1016.806763168581,
  //     0, 0, 1);
  //   fisheye_distcoe = (cv::Mat_<double>(1, 4) << 0.000413327, -0.017352, 0.0175895, -0.0110053);
  //   cv::Mat inputImage = cv::imread(entry.path());
  //   obtain_relative_Rt(inputImage, fisheye_cammatix, fisheye_distcoe, marker_pairs);
  // }

  // gtsam::Values initial;
  // gtsam::NonlinearFactorGraph graph;
  // gtsam::Vector Vector6(6);
  // Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8;
  // gtsam::noiseModel::Diagonal::shared_ptr priorModel = gtsam::noiseModel::Diagonal::Variances(Vector6);
  // initial.insert(0, gtsam::Pose3(gtsam::Rot3(Eigen::MatrixXd::Identity(3, 3)), gtsam::Point3(Eigen::Vector3d::Zero())));
  // graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(gtsam::Rot3(Eigen::MatrixXd::Identity(3, 3)),
  //                                                            gtsam::Point3(Eigen::Vector3d::Zero())), priorModel));
  // for(int i = 1; i < 8; i++)
  //   initial.insert(i, gtsam::Pose3(gtsam::Rot3(Eigen::MatrixXd::Identity(3, 3)), gtsam::Point3(Eigen::Vector3d::Zero())));
  
  // for(int i = 0; i < marker_pairs.size(); i++)
  // {
  //   gtsam::Rot3 R_sam(marker_pairs[i].delta_R);
  //   gtsam::Point3 t_sam(marker_pairs[i].delta_t);
    
  //   Vector6 << marker_pairs[i].cov(0), marker_pairs[i].cov(1), marker_pairs[i].cov(2),
  //              marker_pairs[i].cov(3), marker_pairs[i].cov(4), marker_pairs[i].cov(5);
  //   gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
  //   gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(marker_pairs[i].id1, marker_pairs[i].id2,
  //     gtsam::Pose3(R_sam, t_sam), odometryNoise));
  //   graph.push_back(factor);
  // }
  
  // gtsam::ISAM2Params parameters;
  // parameters.relinearizeThreshold = 0.01;
  // parameters.relinearizeSkip = 1;
  // gtsam::ISAM2 isam(parameters);
  // isam.update(graph, initial);
  // isam.update();

  // gtsam::Values results = isam.calculateEstimate();
  
  // for(uint i = 0; i < results.size(); i++)
  // {
  //   gtsam::Pose3 pose = results.at(i).cast<gtsam::Pose3>();
  //   cout<<"id "<<i<<endl;
  //   Quaterniond q(pose.rotation().matrix());
  //   Vector3d t = pose.translation();
  //   // cout<<q.w()<<", "<<q.x()<<", "<<q.y()<<", "<<q.z()<<", "<<t(0)<<", "<<t(1)<<", "<<t(2)<<endl;
  //   cout<<"R "<<pose.rotation().matrix()<<endl;
  //   cout<<"t "<<pose.translation()<<endl<<endl;
  // }
}