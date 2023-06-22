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

/* 优化tag系到相机系的外参、相机外参、tag之间外参 */
class single_tag_twin_camera_residual
{
public:
  single_tag_twin_camera_residual(VPnPData p, Camera camera) {p_ = p; camera_ = camera;}

  template <typename T>
  bool operator()(const T* _q, const T* _t,
                  const T* _q1, const T* _t1,
                  const T* _q2, const T* _t2,
                  T* residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = camera_.intrinsic.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = camera_.distortion.cast<T>();

    Eigen::Quaternion<T> q_proj{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_proj{_t[0], _t[1], _t[2]};

    Eigen::Quaternion<T> q_cam1{_q1[3], _q1[0], _q1[1], _q1[2]};
    Eigen::Matrix<T, 3, 1> t_cam1{_t1[0], _t1[1], _t1[2]};

    Eigen::Quaternion<T> q_cam2{_q2[3], _q2[0], _q2[1], _q2[2]};
    Eigen::Matrix<T, 3, 1> t_cam2{_t2[0], _t2[1], _t2[2]};
    
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
  
  static ceres::CostFunction *Create(VPnPData p, Camera camera)
  {
    return (new ceres::NumericDiffCostFunction<single_tag_twin_camera_residual, ceres::CENTRAL, 2, 4, 3, 4, 3, 4, 3>(
      new single_tag_twin_camera_residual(p, camera)));
  }

private:
  VPnPData p_;
  Camera camera_;
};