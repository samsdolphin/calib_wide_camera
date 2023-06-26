#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>

#include <ros/ros.h>

#include <Eigen/Dense>
#include "ceres/ceres.h"
#include "ceres/loss_function.h"

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "ceres_residual.hpp"

// #include <gtsam/geometry/Pose3.h>
// #include <gtsam/slam/PriorFactor.h>
// #include <gtsam/slam/BetweenFactor.h>
// #include <gtsam/nonlinear/Values.h>
// #include <gtsam/nonlinear/ISAM2.h>

// #define CALIB_TAG_ONLY
#define TOTAL_TAG_NUM 8

using namespace cv;
using namespace std;
using namespace Eigen;

int img_num = 0;
ceres::LossFunction* loss_function;

string single_cam_path = "/media/sam/CR7/20230613_shenzhen_rosbag/calib_aruco/";
string twin_cam_path = "/media/sam/CR7/20230613_shenzhen_rosbag/calib_fishcam/";

vector<string> read_filenames(string filepath)
{
  std::vector<std::string> result;
  for(const auto& entry: std::filesystem::directory_iterator(filepath))
    result.push_back(entry.path());
  return result;
}

/* 仅优化tag之间的外参 */
class tag_ext_residual
{
public:
  tag_ext_residual(VPnPData p, Quaterniond q, Vector3d t, Camera camera)
  {p_ = p; _q = q; _t = t; camera_ = camera;}

  template <typename T>
  bool operator()(const T* _tag1, const T* _tag2, T* residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = camera_.intrinsic.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = camera_.distortion.cast<T>();
    Eigen::Quaternion<T> q_proj = _q.cast<T>();
    Eigen::Matrix<T, 3, 1> t_proj = _t.cast<T>();
    Eigen::Matrix<T, 3, 1> p_l(T(p_.p(0)), T(p_.p(1)), T(p_.p(2)));
    
    Eigen::Matrix<T, 3, 3> delta_R = R_theta(_tag1[2]).transpose() * R_theta(_tag2[2]);
    Eigen::Matrix<T, 3, 1> delta_t = R_theta(_tag1[2]).transpose() * (Eigen::Matrix<T, 3, 1>(_tag2[0]-_tag1[0], _tag2[1]-_tag1[1], T(0)));
    p_l = delta_R * p_l + delta_t;

    Eigen::Matrix<T, 3, 1> p_c = q_proj.toRotationMatrix() * p_l + t_proj;
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
    residuals[0] = ud - T(p_.u);
    residuals[1] = vd - T(p_.v);

    return true;
  }
  
  static ceres::CostFunction *Create(VPnPData p, Quaterniond q, Vector3d t, Camera camera)
  {
    return (new ceres::NumericDiffCostFunction<tag_ext_residual, ceres::CENTRAL, 2, 3, 3>(
      new tag_ext_residual(p, q, t, camera)));
  }

private:
  VPnPData p_;
  Quaterniond _q;
  Vector3d _t;
  Camera camera_;
};

/* 优化tag系到相机系的外参和tag之间的外参 */
class tag_cam_ext_residual
{
public:
  tag_cam_ext_residual(VPnPData p, Camera camera) {p_ = p; camera_ = camera;}

  template <typename T>
  bool operator()(const T* _q, const T* _t, const T* _tag1, const T* _tag2, T* residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = camera_.intrinsic.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = camera_.distortion.cast<T>();
    Eigen::Quaternion<T> q_proj{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_proj{_t[0], _t[1], _t[2]};
    Eigen::Matrix<T, 3, 1> p_l(T(p_.p(0)), T(p_.p(1)), T(p_.p(2)));
    
    Eigen::Matrix<T, 3, 3> delta_R = R_theta(_tag1[2]).transpose() * R_theta(_tag2[2]);
    Eigen::Matrix<T, 3, 1> delta_t = R_theta(_tag1[2]).transpose() * (Eigen::Matrix<T, 3, 1>(_tag2[0]-_tag1[0], _tag2[1]-_tag1[1], T(0)));
    p_l = delta_R * p_l + delta_t;

    Eigen::Matrix<T, 3, 1> p_c = q_proj.toRotationMatrix() * p_l + t_proj;
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
    residuals[0] = ud - T(p_.u);
    residuals[1] = vd - T(p_.v);

    return true;
  }
  
  static ceres::CostFunction *Create(VPnPData p, Camera camera)
  {
    return (new ceres::NumericDiffCostFunction<tag_cam_ext_residual, ceres::CENTRAL, 2, 4, 3, 3, 3>(
      new tag_cam_ext_residual(p, camera)));
  }

private:
  VPnPData p_;
  Camera camera_;
};

/* 仅优化tag系到相机系的外参 */
class tag_camera_residual
{
public:
  tag_camera_residual(VPnPData p, Camera camera) {p_ = p; camera_ = camera;}

  template <typename T>
  bool operator()(const T* _q, const T* _t, T* residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = camera_.intrinsic.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = camera_.distortion.cast<T>();
    Eigen::Quaternion<T> q_proj{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_proj{_t[0], _t[1], _t[2]};
    Eigen::Matrix<T, 3, 1> p_l(T(p_.p(0)), T(p_.p(1)), T(p_.p(2)));
    Eigen::Matrix<T, 3, 1> p_c = q_proj.toRotationMatrix() * p_l + t_proj;
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
    residuals[0] = ud - T(p_.u);
    residuals[1] = vd - T(p_.v);

    return true;
  }
  
  static ceres::CostFunction *Create(VPnPData p, Camera camera)
  {
    return (new ceres::AutoDiffCostFunction<tag_camera_residual, 2, 4, 3>(new tag_camera_residual(p, camera)));
  }

private:
  VPnPData p_;
  Camera camera_;
};

/* 优化tag系到相机系的外参、相机外参、tag之间外参 */
class twin_tag_twin_camera_residual
{
public:
  twin_tag_twin_camera_residual(VPnPData p, Camera camera, Eigen::Vector3d tcam1, Eigen::Vector3d tcam2)
  {p_ = p; camera_ = camera; tcam1_ = tcam1; tcam2_ = tcam2;}

  template <typename T>
  bool operator()(const T* _q, const T* _t,
                  const T* _tag1, const T* _tag2,
                  const T* _q1,
                  const T* _q2,
                  T* residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = camera_.intrinsic.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = camera_.distortion.cast<T>();

    Eigen::Quaternion<T> q_proj{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_proj{_t[0], _t[1], _t[2]};

    Eigen::Matrix<T, 3, 3> delta_tag_R = R_theta(_tag1[2]).transpose() * R_theta(_tag2[2]);
    Eigen::Matrix<T, 3, 1> delta_tag_t = R_theta(_tag1[2]).transpose() * (Eigen::Matrix<T, 3, 1>(_tag2[0]-_tag1[0], _tag2[1]-_tag1[1], T(0)));

    Eigen::Quaternion<T> q_cam1{_q1[3], _q1[0], _q1[1], _q1[2]};
    Eigen::Matrix<T, 3, 1> t_cam1 = tcam1_.cast<T>();

    Eigen::Quaternion<T> q_cam2{_q2[3], _q2[0], _q2[1], _q2[2]};
    Eigen::Matrix<T, 3, 1> t_cam2 = tcam2_.cast<T>();
    
    Eigen::Matrix<T, 3, 1> p_l(T(p_.p(0)), T(p_.p(1)), T(p_.p(2)));
    Eigen::Matrix<T, 3, 1> p_c = q_cam2.inverse() * q_cam1 * q_proj * delta_tag_R * p_l;
    p_c += q_cam2.inverse() * q_cam1 * (q_proj * delta_tag_t + t_proj) + q_cam2.inverse() * (t_cam1 - t_cam2);

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
    residuals[0] = ud - T(p_.u);
    residuals[1] = vd - T(p_.v);

    return true;
  }
  
  static ceres::CostFunction *Create(VPnPData p, Camera camera, Eigen::Vector3d tcam1, Eigen::Vector3d tcam2)
  {
    return (new ceres::NumericDiffCostFunction<twin_tag_twin_camera_residual, ceres::CENTRAL, 2, 4, 3, 3, 3, 4, 4>(
      new twin_tag_twin_camera_residual(p, camera, tcam1, tcam2)));
  }

private:
  VPnPData p_;
  Camera camera_;
  Eigen::Vector3d tcam1_, tcam2_;
};

/* 优化tag系到相机系的外参、相机外参、tag之间外参 */
class single_tag_twin_camera_residual
{
public:
  single_tag_twin_camera_residual(VPnPData p, Camera camera, Eigen::Vector3d tcam1, Eigen::Vector3d tcam2)
  {p_ = p; camera_ = camera; tcam1_ = tcam1; tcam2_ = tcam2;}

  template <typename T>
  bool operator()(const T* _q, const T* _t,
                  const T* _q1,
                  const T* _q2,
                  T* residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = camera_.intrinsic.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = camera_.distortion.cast<T>();

    Eigen::Quaternion<T> q_proj{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_proj{_t[0], _t[1], _t[2]};

    Eigen::Quaternion<T> q_cam1{_q1[3], _q1[0], _q1[1], _q1[2]};
    Eigen::Matrix<T, 3, 1> t_cam1 = tcam1_.cast<T>();

    Eigen::Quaternion<T> q_cam2{_q2[3], _q2[0], _q2[1], _q2[2]};
    Eigen::Matrix<T, 3, 1> t_cam2 = tcam2_.cast<T>();
    
    Eigen::Matrix<T, 3, 1> p_l(T(p_.p(0)), T(p_.p(1)), T(p_.p(2)));
    Eigen::Matrix<T, 3, 1> p_c = q_cam2.inverse() * q_cam1 * q_proj * p_l;
    p_c += q_cam2.inverse() * q_cam1 * t_proj + q_cam2.inverse() * (t_cam1 - t_cam2);

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
    residuals[0] = ud - T(p_.u);
    residuals[1] = vd - T(p_.v);

    return true;
  }
  
  static ceres::CostFunction *Create(VPnPData p, Camera camera, Eigen::Vector3d tcam1, Eigen::Vector3d tcam2)
  {
    return (new ceres::NumericDiffCostFunction<single_tag_twin_camera_residual, ceres::CENTRAL, 2, 4, 3, 4, 4>(
      new single_tag_twin_camera_residual(p, camera, tcam1, tcam2)));
  }

private:
  VPnPData p_;
  Camera camera_;
  Eigen::Vector3d tcam1_, tcam2_;
};

/*每个tag和tag0的外参*/
double tag_ext[3*TOTAL_TAG_NUM];
double img_ext[7*31]; // 前16是单相机位姿，后16是相机对的位姿
double cam_ext[7*4];

void debug_camera(cv::Mat inputImage, Camera camera)
{
  cv::Mat outputImage = inputImage.clone();
  cv::Mat& fisheye_cammatix = camera.cv_intrinsic;
  cv::Mat& fisheye_distcoe = camera.cv_distortion;

  /* Rought Estimation */
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
  cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
  cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
  cv::aruco::detectMarkers(outputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

  // if(markerIds.size() > 0)
  //   cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);
  // cv::imwrite(single_cam_path + to_string(img_num++) +".png", outputImage);

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
      cost_function = tag_camera_residual::Create(vpnp_data[j], camera);
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

    // cout<<"marker id "<<markerIds[i]<<endl;
    // cout<<"final R"<<endl;
    // cout<<m_q.toRotationMatrix()<<endl;
    // cout<<"final t"<<endl;
    // cout<<m_t.transpose()<<endl;

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
      cv::circle(outputImage, pts_2d[j], 3, cv::Scalar(0, 0, 255), -1);
      res += cv::norm(markerCorners[i][j] - pts_2d[j]);
    }
  }
  res /= markerCorners.size()*4;
  // cout<<"average reprojection error (pixel): "<<res<<endl;
  cv::imwrite(single_cam_path + "debug.png", outputImage);
}

void init_tag_extrinsic(cv::Mat inputImage, Camera camera)
{
  cv::Mat outputImage = inputImage.clone();
  cv::Mat& fisheye_cammatix = camera.cv_intrinsic;
  cv::Mat& fisheye_distcoe = camera.cv_distortion;

  /* Rought Estimation */
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
  cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
  cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
  cv::aruco::detectMarkers(outputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

  // if(markerIds.size() > 0)
  //   cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);
  // cv::imwrite(single_cam_path + to_string(img_num++) +".png", outputImage);

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
      cost_function = tag_camera_residual::Create(vpnp_data[j], camera);
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
      // cv::circle(outputImage, pts_2d[j], 3, cv::Scalar(0, 0, 255), -1);
      res += cv::norm(markerCorners[i][j] - pts_2d[j]);
    }
  }
  res /= markerCorners.size()*4;
  // cout<<"average reprojection error (pixel): "<<res<<endl;
  // cv::imwrite(single_cam_path + to_string(img_num++) + "_" + to_string(res) + ".png", outputImage);

  for(int i = 0; i < TOTAL_TAG_NUM; i++)
  {
    Eigen::Matrix3d R0(final_qs[0]);
    Eigen::Vector3d t0 = final_ts[0];
    Eigen::Vector3d t_ = R0.transpose()*(final_ts[i]-t0);
    Eigen::Matrix3d R_ = R0.transpose()*final_qs[i].toRotationMatrix();
    tag_ext[markerIds[i]*3] = t_(0);
    tag_ext[markerIds[i]*3+1] = t_(1);
    tag_ext[markerIds[i]*3+2] = atan2(R_(1, 0), R_(0, 0));
  }
  ROS_INFO_STREAM("tag extrinsics initialized");
}

MarkerPair add_residual(cv::Mat inputImage, Camera camera, ceres::Problem& problem_)
{
  cv::Mat outputImage = inputImage.clone();
  cv::Mat& fisheye_cammatix = camera.cv_intrinsic;
  cv::Mat& fisheye_distcoe = camera.cv_distortion;

  /* Rought Estimation */
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
  cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
  cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
  cv::aruco::detectMarkers(outputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

  std::vector<cv::Vec3d> rvecs, tvecs;
  Matrix3d rotation;
  Vector3d translation;
  cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.15, fisheye_cammatix, fisheye_distcoe, rvecs, tvecs);

  Quaterniond init_qs, final_qs; // 只保存marker_id_0的外参
  Vector3d init_ts, final_ts;
  
  cv::Matx33d rotationMatrix;
  cv::Rodrigues(rvecs[0], rotationMatrix);

  for(int j = 0; j < 3; j++)
  {
    for(int k = 0; k < 3; k++)
      rotation(j, k) = rotationMatrix(j, k);
    translation(j) = tvecs[0](j);
  }
  init_qs = Quaterniond(rotation);
  init_ts = Vector3d(translation(0), translation(1), translation(2));

  /* 优化marker_id_0到相机的外参 */
  double res = 0;
  for(int i = 0; i < 1; i++)
  {
    double ext[7] = {init_qs.x(), init_qs.y(), init_qs.z(), init_qs.w(), init_ts(0), init_ts(1), init_ts(2)}; // qx, qy, qz, qw, tx, ty, tz

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
      cost_function = tag_camera_residual::Create(vpnp_data[j], camera);
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

    final_qs = m_q;
    final_ts = m_t;

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
      // cv::circle(outputImage, pts_2d[j], 3, cv::Scalar(0, 0, 255), -1);
      res += cv::norm(markerCorners[i][j] - pts_2d[j]);
    }
  }
  // res /= markerCorners.size()*4;
  // cout<<"image "<<img_num<<" average reprojection error (pixel): "<<res<<endl;
  // cv::imwrite(single_cam_path + to_string(img_num) + "_" + to_string(res) + ".png", outputImage);

  /*这张图片和相机外参，参考tag是识别的第一个tag*/
  cout<<"init "<<img_num<<"th img extrinsic, detected corners: "<<markerCorners.size()<<endl;
  img_ext[7*img_num+0] = final_qs.x();
  img_ext[7*img_num+1] = final_qs.y();
  img_ext[7*img_num+2] = final_qs.z();
  img_ext[7*img_num+3] = final_qs.w();
  img_ext[7*img_num+4] = final_ts(0);
  img_ext[7*img_num+5] = final_ts(1);
  img_ext[7*img_num+6] = final_ts(2);

  int ref_id = markerIds[0]; // 参考tag的id

  #ifndef CALIB_TAG_ONLY
  ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
  problem_.AddParameterBlock(img_ext+7*img_num, 4, q_parameterization);
  problem_.AddParameterBlock(img_ext+7*img_num+4, 3);
  #endif

  std::vector<std::vector<cv::Point2f>> filtered_corners;
  vector<int> filtered_ids;

  for(int i = 0; i < markerCorners.size(); i++)
  {
    if(markerIds[i] >= TOTAL_TAG_NUM) continue;

    vector<VPnPData> vpnp_data(4);
    vpnp_data[0].p = Eigen::Vector3d(0, 0, 0); vpnp_data[0].u = markerCorners[i][0].x; vpnp_data[0].v = markerCorners[i][0].y;
    vpnp_data[1].p = Eigen::Vector3d(0.15, 0, 0); vpnp_data[1].u = markerCorners[i][1].x; vpnp_data[1].v = markerCorners[i][1].y;
    vpnp_data[2].p = Eigen::Vector3d(0.15, -0.15, 0); vpnp_data[2].u = markerCorners[i][2].x; vpnp_data[2].v = markerCorners[i][2].y;
    vpnp_data[3].p = Eigen::Vector3d(0, -0.15, 0); vpnp_data[3].u = markerCorners[i][3].x; vpnp_data[3].v = markerCorners[i][3].y;

    if(markerIds[i] == ref_id)
      for(int j = 0; j < 4; j++)
      {
        #ifndef CALIB_TAG_ONLY
        ceres::CostFunction *cost_function;
        cost_function = tag_camera_residual::Create(vpnp_data[j], camera);
        problem_.AddResidualBlock(cost_function, new ceres::HuberLoss(0.1), img_ext+7*img_num, img_ext+7*img_num+4);
        #endif
      }
    else
    {
      int cur_id = markerIds[i];
      for(int j = 0; j < 4; j++)
      {
        ceres::CostFunction *cost_function;
        #ifdef CALIB_TAG_ONLY
        cost_function = tag_ext_residual::Create(vpnp_data[j],
                                                 TagData{tag_ext[ref_id*3], tag_ext[ref_id*3+1], tag_ext[ref_id*3+2]},
                                                 TagData{tag_ext[cur_id*3], tag_ext[cur_id*3+1], tag_ext[cur_id*3+2]},
                                                 final_qs[0], final_ts[0], camera);
        problem_.AddResidualBlock(cost_function, new ceres::HuberLoss(0.1), tag_ext + 3*ref_id, tag_ext + 3*cur_id);
        #else
        cost_function = tag_cam_ext_residual::Create(vpnp_data[j], camera);
        problem_.AddResidualBlock(cost_function, new ceres::HuberLoss(0.1),
                                  img_ext+7*img_num, img_ext+7*img_num+4, tag_ext+3*ref_id, tag_ext+3*cur_id);
        #endif
      }
    }
    filtered_corners.push_back(markerCorners[i]);
    filtered_ids.push_back(markerIds[i]);
  }

  img_num++;
  return MarkerPair(img_num-1, ref_id, filtered_ids, filtered_corners);
}

MarkerPair add_twin_residual(cv::Mat img1, cv::Mat img2, Camera camera1, Camera camera2, ceres::Problem& problem_, bool debug = false)
{
  cv::Mat outputImage = img1.clone();
  cv::Mat& fisheye_cammatix1 = camera1.cv_intrinsic;
  cv::Mat& fisheye_distcoe1 = camera1.cv_distortion;

  /* Rought Estimation */
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
  cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
  cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
  cv::aruco::detectMarkers(outputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

  std::vector<cv::Vec3d> rvecs, tvecs;
  Matrix3d rotation;
  Vector3d translation;
  cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.15, fisheye_cammatix1, fisheye_distcoe1, rvecs, tvecs);

  Quaterniond init_qs, final_qs; // 只保存marker_id_0的外参
  Vector3d init_ts, final_ts;

  int ref_id = -1; // 参考tag的id
  int i_idx = -1;
  for(int i = 0; i < markerIds.size(); i++)
  {
    if(markerIds[i] >= TOTAL_TAG_NUM) continue;
    ref_id = markerIds[i];
    i_idx = i;
    break;
  }
  
  cv::Matx33d rotationMatrix;
  cv::Rodrigues(rvecs[i_idx], rotationMatrix);

  for(int j = 0; j < 3; j++)
  {
    for(int k = 0; k < 3; k++)
      rotation(j, k) = rotationMatrix(j, k);
    translation(j) = tvecs[i_idx](j);
  }
  init_qs = Quaterniond(rotation);
  init_ts = Vector3d(translation(0), translation(1), translation(2));

  /* 优化marker_id_0到相机的外参 */
  double res = 0;
  
  if(1)
  {
    int i = i_idx;

    double ext[7] = {init_qs.x(), init_qs.y(), init_qs.z(), init_qs.w(), init_ts(0), init_ts(1), init_ts(2)}; // qx, qy, qz, qw, tx, ty, tz

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
      cost_function = tag_camera_residual::Create(vpnp_data[j], camera1);
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

    final_qs = m_q;
    final_ts = m_t;

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
    cv::fisheye::projectPoints(pts_3d, pts_2d, rvec, tvec, fisheye_cammatix1, fisheye_distcoe1);
    for(int j = 0; j < pts_2d.size(); j++)
    {
      // cv::circle(outputImage, pts_2d[j], 3, cv::Scalar(0, 0, 255), -1);
      res += cv::norm(markerCorners[i][j] - pts_2d[j]);
    }
  }
  // res /= markerCorners.size()*4;
  // cout<<"image "<<img_num<<" average reprojection error (pixel): "<<res<<endl;
  // cout<<"ref id "<<ref_id<<endl;
  // cv::imwrite(single_cam_path + to_string(img_num) + "_" + to_string(res) + ".png", outputImage);

  /* 这张图片和相机外参，参考tag是识别的第一个tag */
  cout<<"init "<<img_num<<"th img extrinsic, detected corners: "<<markerCorners.size()<<" + ";
  img_ext[7*img_num+0] = final_qs.x();
  img_ext[7*img_num+1] = final_qs.y();
  img_ext[7*img_num+2] = final_qs.z();
  img_ext[7*img_num+3] = final_qs.w();
  img_ext[7*img_num+4] = final_ts(0);
  img_ext[7*img_num+5] = final_ts(1);
  img_ext[7*img_num+6] = final_ts(2);

  #ifndef CALIB_TAG_ONLY
  ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
  problem_.AddParameterBlock(img_ext+7*img_num, 4, q_parameterization);
  problem_.AddParameterBlock(img_ext+7*img_num+4, 3);
  #endif

  std::vector<std::vector<cv::Point2f>> filtered_corners;
  vector<int> filtered_ids;

  for(int i = 0; i < markerCorners.size(); i++)
  {
    if(markerIds[i] >= TOTAL_TAG_NUM) continue;
    vector<VPnPData> vpnp_data(4);
    vpnp_data[0].p = Eigen::Vector3d(0, 0, 0); vpnp_data[0].u = markerCorners[i][0].x; vpnp_data[0].v = markerCorners[i][0].y;
    vpnp_data[1].p = Eigen::Vector3d(0.15, 0, 0); vpnp_data[1].u = markerCorners[i][1].x; vpnp_data[1].v = markerCorners[i][1].y;
    vpnp_data[2].p = Eigen::Vector3d(0.15, -0.15, 0); vpnp_data[2].u = markerCorners[i][2].x; vpnp_data[2].v = markerCorners[i][2].y;
    vpnp_data[3].p = Eigen::Vector3d(0, -0.15, 0); vpnp_data[3].u = markerCorners[i][3].x; vpnp_data[3].v = markerCorners[i][3].y;

    if(markerIds[i] == ref_id)
      for(int j = 0; j < 4; j++)
      {
        #ifndef CALIB_TAG_ONLY
        ceres::CostFunction *cost_function;
        cost_function = tag_camera_residual::Create(vpnp_data[j], camera1);
        problem_.AddResidualBlock(cost_function, new ceres::HuberLoss(0.1), img_ext+7*img_num, img_ext+7*img_num+4);
        #endif
      }
    else
    {
      int cur_id = markerIds[i];
      for(int j = 0; j < 4; j++)
      {
        ceres::CostFunction *cost_function;
        #ifdef CALIB_TAG_ONLY
        cost_function = tag_ext_residual::Create(vpnp_data[j],
                                                 TagData{tag_ext[ref_id*3], tag_ext[ref_id*3+1], tag_ext[ref_id*3+2]},
                                                 TagData{tag_ext[cur_id*3], tag_ext[cur_id*3+1], tag_ext[cur_id*3+2]},
                                                 final_qs[0], final_ts[0], camera1);
        problem_.AddResidualBlock(cost_function, new ceres::HuberLoss(0.1), tag_ext + 3*ref_id, tag_ext + 3*cur_id);
        #else
        cost_function = tag_cam_ext_residual::Create(vpnp_data[j], camera1);
        problem_.AddResidualBlock(cost_function, new ceres::HuberLoss(0.1),
                                  img_ext+7*img_num, img_ext+7*img_num+4, tag_ext+3*ref_id, tag_ext+3*cur_id);
        #endif
      }
    }
    filtered_corners.push_back(markerCorners[i]);
    filtered_ids.push_back(markerIds[i]);
  }

  cv::Mat outputImage2 = img2.clone();
  std::vector<int> markerIds2;
  std::vector<std::vector<cv::Point2f>> markerCorners2, rejectedCandidates2;
  cv::Ptr<cv::aruco::DetectorParameters> parameters2 = cv::aruco::DetectorParameters::create();
  cv::Ptr<cv::aruco::Dictionary> dictionary2 = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
  cv::aruco::detectMarkers(outputImage2, dictionary2, markerCorners2, markerIds2, parameters2, rejectedCandidates2);
  cout<<markerCorners2.size()<<endl;

  std::vector<std::vector<cv::Point2f>> filtered_corners2;
  vector<int> filtered_ids2;

  for(int i = 0; i < markerCorners2.size(); i++)
  {
    if(markerIds2[i] >= TOTAL_TAG_NUM) continue;

    vector<VPnPData> vpnp_data(4);
    vpnp_data[0].p = Eigen::Vector3d(0, 0, 0); vpnp_data[0].u = markerCorners2[i][0].x; vpnp_data[0].v = markerCorners2[i][0].y;
    vpnp_data[1].p = Eigen::Vector3d(0.15, 0, 0); vpnp_data[1].u = markerCorners2[i][1].x; vpnp_data[1].v = markerCorners2[i][1].y;
    vpnp_data[2].p = Eigen::Vector3d(0.15, -0.15, 0); vpnp_data[2].u = markerCorners2[i][2].x; vpnp_data[2].v = markerCorners2[i][2].y;
    vpnp_data[3].p = Eigen::Vector3d(0, -0.15, 0); vpnp_data[3].u = markerCorners2[i][3].x; vpnp_data[3].v = markerCorners2[i][3].y;

    if(markerIds2[i] == ref_id)
      for(int j = 0; j < 4; j++)
      {
        ceres::CostFunction *cost_function;
        Eigen::Vector3d tcam1(cam_ext[camera1.id*7+4], cam_ext[camera1.id*7+5], cam_ext[camera1.id*7+6]);
        Eigen::Vector3d tcam2(cam_ext[camera2.id*7+4], cam_ext[camera2.id*7+5], cam_ext[camera2.id*7+6]);
        cost_function = single_tag_twin_camera_residual::Create(vpnp_data[j], camera2, tcam1, tcam2);
        problem_.AddResidualBlock(cost_function, new ceres::HuberLoss(0.1),
                                  img_ext+7*img_num, img_ext+7*img_num+4, // tag系到cam1系的外参
                                  cam_ext+camera1.id*7, // cam1的R
                                  cam_ext+camera2.id*7); // cam2的R
      }
    else
    {
      int cur_id = markerIds2[i];
      for(int j = 0; j < 4; j++)
      {
        ceres::CostFunction *cost_function;
        Eigen::Vector3d tcam1(cam_ext[camera1.id*7+4], cam_ext[camera1.id*7+5], cam_ext[camera1.id*7+6]);
        Eigen::Vector3d tcam2(cam_ext[camera2.id*7+4], cam_ext[camera2.id*7+5], cam_ext[camera2.id*7+6]);
        cost_function = twin_tag_twin_camera_residual::Create(vpnp_data[j], camera2, tcam1, tcam2);
        problem_.AddResidualBlock(cost_function, new ceres::HuberLoss(0.1),
                                  img_ext+7*img_num, img_ext+7*img_num+4, // tag系到cam1系的外参
                                  tag_ext+3*ref_id, // ref_tag的外参
                                  tag_ext+3*cur_id, // 当前tag的外参
                                  cam_ext+camera1.id*7, // cam1的位姿
                                  cam_ext+camera2.id*7); // cam2的位姿
      }
    }
    filtered_corners2.push_back(markerCorners2[i]);
    filtered_ids2.push_back(markerIds2[i]);
  }

  img_num++;
  return MarkerPair(img_num-1, ref_id, filtered_ids2, filtered_corners2, camera1.id, camera2.id);
}

void calculate_reprojection_err(cv::Mat inputImage, Camera camera, MarkerPair marker_pair, double& proj_err)
{
  cv::Mat outputImage = inputImage.clone();
  int img_id = marker_pair.image_id;
  int ref_id = marker_pair.ref_tag_id;
  vector<int>& markerIds = marker_pair.marker_ids;
  vector<vector<cv::Point2f>>& markerCorners = marker_pair.marker_corners;
  cv::Mat& fisheye_cammatix = camera.cv_intrinsic;
  cv::Mat& fisheye_distcoe = camera.cv_distortion;

  /* 验证优化后的初始值重投影误差 */
  Eigen::Matrix3d R0 = R_theta(tag_ext[ref_id*3+2]);
  Eigen::Vector3d t0(tag_ext[ref_id*3], tag_ext[ref_id*3+1], 0);

  Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(img_ext+7*img_id);
  Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(img_ext+7*img_id + 4);

  // cout<<"here "<<m_q.x()<<" "<<m_q.y()<<" "<<m_q.z()<<" "<<m_q.w()<<endl;
  // cout<<"here "<<m_t(0)<<" "<<m_t(1)<<" "<<m_t(2)<<endl;

  double res = 0;
  for(int i = 0; i < markerIds.size(); i++)
  {
    int cur_id = markerIds[i];
    // cout<<"cur id "<<cur_id<<endl;
    Eigen::Matrix3d new_R = R0.transpose()*R_theta(tag_ext[cur_id*3+2]);
    Eigen::Vector3d new_t = R0.transpose()*(Eigen::Vector3d(tag_ext[cur_id*3], tag_ext[cur_id*3+1], 0)-t0);
    // cout<<"new_R "<<new_R<<endl;
    // cout<<"new_t "<<new_t<<endl;

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
      cv::circle(outputImage, pts_2d[j], 3, cv::Scalar(0, 0, 255), -1);
      res += cv::norm(markerCorners[i][j] - pts_2d[j]);
    }
  }
  res /= markerCorners.size()*4;
  proj_err += res;
  cout<<"img "<<img_id<<" avg re-proj err (pixel): "<<res<<endl;
  cv::imwrite(single_cam_path + "reproj_" + to_string(img_id) + ".png", outputImage);
}

void calculate_twin_reprojection_err(cv::Mat inputImage, Camera camera, MarkerPair marker_pair, double& proj_err)
{
  cv::Mat outputImage = inputImage.clone(); // camera2的图片
  int img_id = marker_pair.image_id;
  int ref_id = marker_pair.ref_tag_id;
  int cam1_id = marker_pair.cam_id1;
  int cam2_id = marker_pair.cam_id2;
  vector<int>& markerIds = marker_pair.marker_ids;
  vector<vector<cv::Point2f>>& markerCorners = marker_pair.marker_corners;
  cv::Mat& fisheye_cammatix = camera.cv_intrinsic;
  cv::Mat& fisheye_distcoe = camera.cv_distortion;

  /* 验证优化后的初始值重投影误差 */
  Eigen::Matrix3d R0 = R_theta(tag_ext[ref_id*3+2]);
  Eigen::Vector3d t0(tag_ext[ref_id*3], tag_ext[ref_id*3+1], 0);

  Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(img_ext+7*img_id); // camera1的投影矩阵
  Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(img_ext+7*img_id + 4);

  Eigen::Map<Eigen::Quaterniond> q_cam1 = Eigen::Map<Eigen::Quaterniond>(cam_ext+7*cam1_id); // camera1的外参
  Eigen::Map<Eigen::Vector3d> t_cam1 = Eigen::Map<Eigen::Vector3d>(cam_ext+7*cam1_id + 4);

  Eigen::Map<Eigen::Quaterniond> q_cam2 = Eigen::Map<Eigen::Quaterniond>(cam_ext+7*cam2_id); // camera2的外参
  Eigen::Map<Eigen::Vector3d> t_cam2 = Eigen::Map<Eigen::Vector3d>(cam_ext+7*cam2_id + 4);

  Eigen::Matrix3d delta_cam_R = q_cam2.toRotationMatrix().transpose() * q_cam1.toRotationMatrix();
  Eigen::Vector3d delta_cam_t = q_cam2.toRotationMatrix().transpose() * (t_cam1 - t_cam2);

  double res = 0;
  for(int i = 0; i < markerIds.size(); i++)
  {
    int cur_id = markerIds[i];
    Eigen::Matrix3d delta_tag_R = R0.transpose()*R_theta(tag_ext[cur_id*3+2]);
    Eigen::Vector3d delta_tag_t = R0.transpose()*(Eigen::Vector3d(tag_ext[cur_id*3], tag_ext[cur_id*3+1], 0)-t0);

    vector<cv::Point3f> pts_3d;
    pts_3d.push_back(cv::Point3f(0, 0, 0));
    pts_3d.push_back(cv::Point3f(0.15, 0, 0));
    pts_3d.push_back(cv::Point3f(0.15, -0.15, 0));
    pts_3d.push_back(cv::Point3f(0, -0.15, 0));
    
    std::vector<cv::Point2f> pts_2d;
    Eigen::Matrix3d rotation = delta_cam_R * m_q.toRotationMatrix() * delta_tag_R;
    Eigen::Vector3d my_t = delta_cam_R * (m_q.toRotationMatrix() * delta_tag_t + m_t) + delta_cam_t;

    cv::Mat R = (cv::Mat_<double>(3,3) <<
      rotation(0, 0), rotation(0, 1), rotation(0, 2),
      rotation(1, 0), rotation(1, 1), rotation(1, 2),
      rotation(2, 0), rotation(2, 1), rotation(2, 2));
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    cv::Mat tvec = (cv::Mat_<double>(3, 1) << my_t(0), my_t(1), my_t(2));
    cv::fisheye::projectPoints(pts_3d, pts_2d, rvec, tvec, fisheye_cammatix, fisheye_distcoe);
    for(int j = 0; j < pts_2d.size(); j++)
    {
      cv::circle(outputImage, pts_2d[j], 3, cv::Scalar(0, 0, 255), -1);
      res += cv::norm(markerCorners[i][j] - pts_2d[j]);
    }
  }
  res /= markerCorners.size()*4;
  proj_err += res;
  cout<<"img "<<img_id<<" avg re-proj err (pixel): "<<res<<endl;
  cv::imwrite(single_cam_path + "reproj_" + to_string(img_id) + "_" + to_string(cam2_id) + ".png", outputImage);
}

void init_camera_intrinsics(vector<Camera>& cameras)
{
  cameras[0].id = 0;
  cameras[0].intrinsic <<
    1418.361670309597, 0, 1253.973935943561,
    0, 1419.101900308538, 1036.466628861505,
    0, 0, 1;
  cameras[0].distortion << -8.45658e-05, -0.00619387, -0.00286654, 0.00127071; // FR
  cameras[0].cv_intrinsic = (cv::Mat_<double>(3, 3) <<
    1418.361670309597, 0, 1253.973935943561,
    0, 1419.101900308538, 1036.466628861505,
    0, 0, 1);
  cameras[0].cv_distortion = (cv::Mat_<double>(1, 4) << -8.45658e-05, -0.00619387, -0.00286654, 0.00127071); // FR

  cameras[1].id = 1;
  cameras[1].intrinsic <<
    1420.153047506825, 0, 1248.175743932881,
    0, 1421.026042801145, 1016.806763168581,
    0, 0, 1;
  cameras[1].distortion << 0.000413327, -0.017352, 0.0175895, -0.0110053; // BL
  cameras[1].cv_intrinsic = (cv::Mat_<double>(3, 3) <<
    1420.153047506825, 0, 1248.175743932881,
    0, 1421.026042801145, 1016.806763168581,
    0, 0, 1);
  cameras[1].cv_distortion = (cv::Mat_<double>(1, 4) << 0.000413327, -0.017352, 0.0175895, -0.0110053); // BL

  cameras[2].id = 2;
  cameras[2].intrinsic <<
    1420.341348618206, 0, 1224.37438458383,
    0, 1420.997567384703, 1010.762813735306,
    0, 0, 1;
  cameras[2].distortion << -0.00425799, 0.00307152, -0.0155525, 0.00805682; // FL
  cameras[2].cv_intrinsic = (cv::Mat_<double>(3, 3) <<
    1420.341348618206, 0, 1224.37438458383,
    0, 1420.997567384703, 1010.762813735306,
    0, 0, 1);
  cameras[2].cv_distortion = (cv::Mat_<double>(1, 4) << -0.00425799, 0.00307152, -0.0155525, 0.00805682); // FL

  cameras[3].id = 3;
  cameras[3].intrinsic <<
    1418.771097125488, 0, 1212.215584588221,
    0, 1419.5068407428, 1042.056348573678,
    0, 0, 1;
  cameras[3].distortion << -0.00362874, 0.00406696, -0.0204213, 0.0122873; // BR
  cameras[3].cv_intrinsic = (cv::Mat_<double>(3, 3) <<
    1418.771097125488, 0, 1212.215584588221,
    0, 1419.5068407428, 1042.056348573678,
    0, 0, 1);
  cameras[3].cv_distortion = (cv::Mat_<double>(1, 4) << -0.00362874, 0.00406696, -0.0204213, 0.0122873); // BR
  ROS_INFO_STREAM("camera intrinsics initialized");
}

void init_camera_extrinsics()
{
  // qx, qy, qz, qw, tx, ty, tz
  cam_ext[0] = 0; cam_ext[1] = 0; cam_ext[2] = 0; cam_ext[3] = 1; cam_ext[4] = 0; cam_ext[5] = 0; cam_ext[6] = 0; // FR
  cam_ext[7] = 0; cam_ext[8] = -0.707107; cam_ext[9] = 0; cam_ext[10] = 0.707107; cam_ext[11] = -0.0561; cam_ext[12] = 0; cam_ext[13] = -0.0315; // BL
  cam_ext[14] = 0; cam_ext[15] = 1; cam_ext[16] = 0; cam_ext[17] = 0; cam_ext[18] = 0.0695; cam_ext[19] = 0; cam_ext[20] = -0.2021; // FL
  cam_ext[21] = 0; cam_ext[22] = 0.707107; cam_ext[23] = 0; cam_ext[24] = 0.707107; cam_ext[25] = 0.1951; cam_ext[26] = 0; cam_ext[27] = -0.06; // BR
  ROS_INFO_STREAM("camera extrinsics initialized");
}

int main()
{
  vector<Camera> cameras(4); // FR, BL, FL, BR
  init_camera_intrinsics(cameras);
  init_camera_extrinsics();
  cv::Mat inputImage = cv::imread(single_cam_path + "fr/3.png");
  init_tag_extrinsic(inputImage, cameras[0]);

  ceres::Problem problem;
  loss_function = new ceres::HuberLoss(0.1);
  for(int i = 0; i < TOTAL_TAG_NUM; i++)
    problem.AddParameterBlock(tag_ext + 3*i, 3);
  for(int i = 0; i < 4; i++)
  {
    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
    problem.AddParameterBlock(cam_ext+7*i, 4, q_parameterization);
    // problem.AddParameterBlock(cam_ext+7*i+4, 3);
  }
  if(problem.HasParameterBlock(tag_ext))
    problem.SetParameterBlockConstant(tag_ext); // 第一个tag的位姿固定
  if(problem.HasParameterBlock(cam_ext))
    problem.SetParameterBlockConstant(cam_ext); // 第一个相机的位姿固定    
  
  vector<MarkerPair> marker_pairs, twin_marker_pairs;

  cout<<"add fr single residual"<<endl;
  vector<string> fr_img_names = read_filenames(single_cam_path + "fr/");
  for(int i = 0; i < fr_img_names.size(); i++)
  {
    inputImage = cv::imread(fr_img_names[i]);
    MarkerPair marker_pair = add_residual(inputImage, cameras[0], problem);
    marker_pairs.push_back(marker_pair);
  }

  cout<<"add bl single residual"<<endl;
  vector<string> bl_img_names = read_filenames(single_cam_path + "bl/");
  for(int i = 0; i < bl_img_names.size(); i++)
  {
    inputImage = cv::imread(bl_img_names[i]);
    MarkerPair marker_pair = add_residual(inputImage, cameras[1], problem);
    marker_pairs.push_back(marker_pair);
  }

  cout<<"add fl single residual"<<endl;
  vector<string> fl_img_names = read_filenames(single_cam_path + "fl/");
  for(int i = 0; i < fl_img_names.size(); i++)
  {
    inputImage = cv::imread(fl_img_names[i]);
    MarkerPair marker_pair = add_residual(inputImage, cameras[2], problem);
    marker_pairs.push_back(marker_pair);
  }

  cout<<"add br single residual"<<endl;
  vector<string> br_img_names = read_filenames(single_cam_path + "br/");
  for(int i = 0; i < br_img_names.size(); i++)
  {
    inputImage = cv::imread(br_img_names[i]);
    MarkerPair marker_pair = add_residual(inputImage, cameras[3], problem);
    marker_pairs.push_back(marker_pair);
  }

  cout<<"add fr-bl pair residual"<<endl;
  vector<string> frbl_img_names = read_filenames(twin_cam_path + "fr_bl/fr/");
  vector<string> blfr_img_names = read_filenames(twin_cam_path + "fr_bl/bl/");
  for(int i = 0; i < frbl_img_names.size(); i++)
  {
    cv::Mat img1 = cv::imread(frbl_img_names[i]);
    cv::Mat img2 = cv::imread(blfr_img_names[i]);
    MarkerPair marker_pair = add_twin_residual(img1, img2, cameras[0], cameras[1], problem);
    twin_marker_pairs.push_back(marker_pair);
  }

  cout<<"add bl-fl pair residual"<<endl;
  vector<string> blfl_img_names = read_filenames(twin_cam_path + "bl_fl/bl/");
  vector<string> flbl_img_names = read_filenames(twin_cam_path + "bl_fl/fl/");
  for(int i = 0; i < blfl_img_names.size(); i++)
  {
    cv::Mat img1 = cv::imread(blfl_img_names[i]);
    cv::Mat img2 = cv::imread(flbl_img_names[i]);
    MarkerPair marker_pair = add_twin_residual(img1, img2, cameras[1], cameras[2], problem);
    twin_marker_pairs.push_back(marker_pair);
  }

  cout<<"add fl-br pair residual"<<endl;
  vector<string> flbr_img_names = read_filenames(twin_cam_path + "fl_br/fl/");
  vector<string> brfl_img_names = read_filenames(twin_cam_path + "fl_br/br/");
  for(int i = 0; i < flbr_img_names.size(); i++)
  {
    cv::Mat img1 = cv::imread(flbr_img_names[i]);
    cv::Mat img2 = cv::imread(brfl_img_names[i]);
    MarkerPair marker_pair = add_twin_residual(img1, img2, cameras[2], cameras[3], problem);
    twin_marker_pairs.push_back(marker_pair);
  }

  cout<<"add br-fr pair residual"<<endl;
  vector<string> brfr_img_names = read_filenames(twin_cam_path + "br_fr/br/");
  vector<string> frbr_img_names = read_filenames(twin_cam_path + "br_fr/fr/");
  for(int i = 0; i < brfr_img_names.size(); i++)
  {
    cv::Mat img1 = cv::imread(brfr_img_names[i]);
    cv::Mat img2 = cv::imread(frbr_img_names[i]);
    MarkerPair marker_pair = add_twin_residual(img1, img2, cameras[3], cameras[0], problem);
    twin_marker_pairs.push_back(marker_pair);
  }
  
  ceres::Solver::Options options;
  options.preconditioner_type = ceres::JACOBI;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = 1;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  double reproj_err = 0;

  int twin_idx = 0;
  for(int i = 0; i < blfr_img_names.size(); i++)
  {
    inputImage = cv::imread(blfr_img_names[i]);
    calculate_twin_reprojection_err(inputImage, cameras[1], twin_marker_pairs[i], reproj_err);
  }
  twin_idx += blfr_img_names.size();

  for(int i = twin_idx; i < flbl_img_names.size()+twin_idx; i++)
  {
    inputImage = cv::imread(flbl_img_names[i-twin_idx]);
    calculate_twin_reprojection_err(inputImage, cameras[2], twin_marker_pairs[i], reproj_err);
  }
  twin_idx += flbl_img_names.size();

  for(int i = twin_idx; i < brfl_img_names.size()+twin_idx; i++)
  {
    inputImage = cv::imread(brfl_img_names[i-twin_idx]);
    calculate_twin_reprojection_err(inputImage, cameras[3], twin_marker_pairs[i], reproj_err);
  }
  twin_idx += brfl_img_names.size();

  for(int i = twin_idx; i < frbr_img_names.size()+twin_idx; i++)
  {
    inputImage = cv::imread(frbr_img_names[i-twin_idx]);
    calculate_twin_reprojection_err(inputImage, cameras[0], twin_marker_pairs[i], reproj_err);
  }

  int idx = 0;
  for(int i = 0; i < fr_img_names.size(); i++)
  {
    inputImage = cv::imread(fr_img_names[i]);
    calculate_reprojection_err(inputImage, cameras[0], marker_pairs[i], reproj_err);
  }
  idx += fr_img_names.size();

  for(int i = idx; i < bl_img_names.size()+idx; i++)
  {
    inputImage = cv::imread(bl_img_names[i-idx]);
    calculate_reprojection_err(inputImage, cameras[1], marker_pairs[i], reproj_err);
  }
  idx += bl_img_names.size();

  for(int i = idx; i < fl_img_names.size()+idx; i++)
  {
    inputImage = cv::imread(fl_img_names[i-idx]);
    calculate_reprojection_err(inputImage, cameras[2], marker_pairs[i], reproj_err);
  }
  idx += fl_img_names.size();

  for(int i = idx; i < br_img_names.size()+idx; i++)
  {
    inputImage = cv::imread(br_img_names[i-idx]);
    calculate_reprojection_err(inputImage, cameras[3], marker_pairs[i], reproj_err);
  }

  cout<<"avg re-proj err "<<reproj_err/31<<endl;

  // Eigen::Matrix3d R0 = R_theta(tag_ext[0*3+2]);
  // Eigen::Vector3d t0(tag_ext[0*3], tag_ext[0*3+1], 0);
  // for(int i = 0; i < TOTAL_TAG_NUM; i++)
  // {
  //   Eigen::Matrix3d new_R = R0.transpose()*R_theta(tag_ext[i*3+2]);
  //   Eigen::Vector3d new_t = R0.transpose()*(Eigen::Vector3d(tag_ext[i*3], tag_ext[i*3+1], 0)-t0);
  //   cout<<"id "<<i<<endl;
  //   cout<<new_R<<endl;
  //   cout<<new_t.transpose()<<endl;
  // }
  Eigen::Map<Eigen::Quaterniond> q_cam1 = Eigen::Map<Eigen::Quaterniond>(cam_ext); // camera1的外参
  Eigen::Map<Eigen::Vector3d> t_cam1 = Eigen::Map<Eigen::Vector3d>(cam_ext + 4);

  for(int i = 1; i < 4; i++)
  {
    Eigen::Map<Eigen::Quaterniond> q_cam2 = Eigen::Map<Eigen::Quaterniond>(cam_ext+7*i); // camera2的外参
    Eigen::Map<Eigen::Vector3d> t_cam2 = Eigen::Map<Eigen::Vector3d>(cam_ext+7*i + 4);

    Eigen::Matrix3d delta_cam_R = q_cam1.toRotationMatrix().transpose() * q_cam2.toRotationMatrix();
    Eigen::Vector3d delta_cam_t = q_cam1.toRotationMatrix().transpose() * (t_cam2 - t_cam1);

    cout<<"cam "<<i<<endl;
    cout<<delta_cam_R(0, 0)<<","<<delta_cam_R(0, 1)<<","<<delta_cam_R(0, 2)<<","<<endl;
    cout<<delta_cam_R(1, 0)<<","<<delta_cam_R(1, 1)<<","<<delta_cam_R(1, 2)<<","<<endl;
    cout<<delta_cam_R(2, 0)<<","<<delta_cam_R(2, 1)<<","<<delta_cam_R(2, 2)<<endl;
    cout<<delta_cam_t(0)<<","<<delta_cam_t(1)<<","<<delta_cam_t(2)<<endl;
  }
  
}