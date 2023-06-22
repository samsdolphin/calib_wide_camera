#include <fstream>
#include <iomanip>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define COLOR_EACH_CAMERA

using namespace std;
using namespace Eigen;
using namespace cv;

void color_cloud(Mat fish_intrinsic, Mat fish_distcoe, Mat input_img,
                 Matrix3d rotation, Vector3d translation,
                 pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_cloud,
                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr& color_cloud)
{
  vector<cv::Point3f> pts_3d;
  for(int i = 0; i < lidar_cloud->points.size(); i++)
  {
    Eigen::Vector3d pt_3d(lidar_cloud->points[i].x, lidar_cloud->points[i].y, lidar_cloud->points[i].z);
    if(pt_3d.norm() > 50 || pt_3d.norm() < 1)
      continue;
    if(rotation.row(2).dot(pt_3d-translation) < 0) continue;
    pts_3d.push_back(Point3f(pt_3d(0), pt_3d(1), pt_3d(2)));
  }

  cv::Mat R = (cv::Mat_<double>(3,3) <<
    rotation(0, 0), rotation(0, 1), rotation(0, 2),
    rotation(1, 0), rotation(1, 1), rotation(1, 2),
    rotation(2, 0), rotation(2, 1), rotation(2, 2));
  cv::Mat rvec;
  cv::Rodrigues(R, rvec);
  cv::Mat tvec = (cv::Mat_<double>(3, 1) << translation(0), translation(1), translation(2));

  std::vector<cv::Point2f> pts_2d;
  cv::fisheye::projectPoints(pts_3d, pts_2d, rvec, tvec, fish_intrinsic, fish_distcoe);

  int image_rows = input_img.rows;
  int image_cols = input_img.cols;
  // color_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
  for(size_t i = 0; i < pts_2d.size(); i++) {
    if(pts_2d[i].x >= 0 && pts_2d[i].x < image_cols && pts_2d[i].y >= 0 && pts_2d[i].y < image_rows)
    {
      
      cv::Scalar color = input_img.at<cv::Vec3b>((int)pts_2d[i].y, (int)pts_2d[i].x);
      if(color[0] == 0 && color[1] == 0 && color[2] == 0)
        continue;
      // if(pts_3d[i].x > 100)
      //   continue;

      pcl::PointXYZRGB p;
      p.x = pts_3d[i].x;
      p.y = pts_3d[i].y;
      p.z = pts_3d[i].z;
      // p.a = 255;
      p.b = color[0];
      p.g = color[1];
      p.r = color[2];
      color_cloud->points.push_back(p);
    }
  }
  color_cloud->width = color_cloud->points.size();
  color_cloud->height = 1;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "validate_fishcam");
  ros::NodeHandle nh("~");

  ros::Publisher pub_fr_cloud = nh.advertise<sensor_msgs::PointCloud2>("/fr_cloud", 100);
  ros::Publisher pub_fl_cloud = nh.advertise<sensor_msgs::PointCloud2>("/fl_cloud", 100);
  ros::Publisher pub_br_cloud = nh.advertise<sensor_msgs::PointCloud2>("/br_cloud", 100);
  ros::Publisher pub_bl_cloud = nh.advertise<sensor_msgs::PointCloud2>("/bl_cloud", 100);

  pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::io::loadPCDFile<pcl::PointXYZI>("/media/sam/CR7/20230613_shenzhen_rosbag/calib_fishcam/lidarcloud.pcd", *lidar_cloud);

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

  Eigen::Matrix3d R_cam_FR;
  R_cam_FR << -0.999993,0.00230445,-0.00295025,
              -0.00295462,-0.00189582,0.999994,
              0.00229884,0.999996,0.00190261;
  Eigen::Vector3d t_cam_FR(0.0695, 0.06885, -0.0765);

  Eigen::Matrix3d R_cam_BR;
  R_cam_BR << -0.00122552,-0.999995,-0.00307221,
              -0.00423693,-0.00306699,0.999986,
              -0.99999,0.00123852,-0.00423315;
  Eigen::Vector3d t_cam_BR(0.0165, 0.06885, -0.1256);

  Eigen::Matrix3d R_cam_BL;
  // R_cam_BL << 0.000562243,0.999999,-0.000832109,
  //             -0.00201167,0.000833239,0.999998,
  //             0.999998,-0.000560567,0.00201213;
  R_cam_BL << -0.000364663,0.999965,-0.00841718,
              -0.00343528,0.00841588,0.999959,
              0.999994,0.000393563,0.00343209;
  Eigen::Vector3d t_cam_BL(0.045, 0.06885, -0.1256);

  Eigen::Matrix3d R_cam_FL;
  R_cam_FL << 0.999992,0.00289236,-0.00268373,
              0.00265804,0.00884739,0.999957,
              0.00291598,-0.999957,0.00883963;
  Eigen::Vector3d t_cam_FL(0, 0.06885, -0.1256);

  /* 重投影误差计算出来的 */
  Eigen::Matrix3d R_FR_BL, R_FR_FL, R_FR_BR;
  R_FR_BL << 0.00646949,0.0016084,-0.999978,
             -0.0108792,0.99994,0.00153795,
             0.999921,0.010869,0.0064866;
  R_FR_FL << -0.999956,-0.00766772,-0.00540209,
             -0.0077316,0.999899,0.0119048,
             0.00531027,0.011946,-0.999915;
  R_FR_BR << -0.00260672,-0.000740075,0.999997,
             0.000377512,1,0.000741061,
             -0.999997,0.000379442,-0.00260644;
  Vector3d t_FR_BL(-0.10509,-0.00111985,-0.171438);
  Vector3d t_FR_FL(0.0647859,0.000803841,-0.291129);
  Vector3d t_FR_BR(0.237676,-0.000321503,-0.107592);

  /* CAD中每个相机到fr的Rt */
  // R_FR_BL << 0, 0, -1, 0, 1, 0, 1, 0, 0;
  // R_FR_FL << -1, 0, 0, 0, 1, 0, 0, 0, -1;
  // R_FR_BR << 0, 0, 1, 0, 1, 0, -1, 0, 0;
  // Vector3d t_FR_BL(-0.0561,0,-0.0315);
  // Vector3d t_FR_FL(0.0695,0,-0.2021);
  // Vector3d t_FR_BR(0.1951,0,-0.06);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  cout<<"push enter"<<endl;
  getchar();
  cv::Mat input_img = cv::imread("/media/sam/CR7/20230613_shenzhen_rosbag/calib_fishcam/fr.png", cv::IMREAD_UNCHANGED);
  color_cloud(fr_intrinsic, fr_coef, input_img, R_cam_FR, t_cam_FR, lidar_cloud, rgb_cloud);

  sensor_msgs::PointCloud2 cloudMsg;
  pcl::toROSMsg(*rgb_cloud, cloudMsg);
  cloudMsg.header.frame_id = "camera_init";
  cloudMsg.header.stamp = ros::Time::now();
  pub_fr_cloud.publish(cloudMsg);

  cout<<"push enter"<<endl;
  getchar();
  input_img = cv::imread("/media/sam/CR7/20230613_shenzhen_rosbag/calib_fishcam/bl.png", cv::IMREAD_UNCHANGED);
  #ifdef COLOR_EACH_CAMERA
  color_cloud(bl_intrinsic, bl_coef, input_img, R_cam_BL, t_cam_BL, lidar_cloud, rgb_cloud);
  #else
  color_cloud(bl_intrinsic, bl_coef, input_img, R_FR_BL.transpose()*R_cam_FR, R_FR_BL.transpose()*(t_cam_FR-t_FR_BL), lidar_cloud, rgb_cloud);
  #endif
  pcl::toROSMsg(*rgb_cloud, cloudMsg);
  cloudMsg.header.frame_id = "camera_init";
  cloudMsg.header.stamp = ros::Time::now();
  pub_bl_cloud.publish(cloudMsg);

  cout<<"push enter"<<endl;
  getchar();
  input_img = cv::imread("/media/sam/CR7/20230613_shenzhen_rosbag/calib_fishcam/br.png", cv::IMREAD_UNCHANGED);
  #ifdef COLOR_EACH_CAMERA
  color_cloud(br_intrinsic, br_coef, input_img, R_cam_BR, t_cam_BR, lidar_cloud, rgb_cloud);
  #else
  color_cloud(br_intrinsic, br_coef, input_img, R_FR_BR.transpose()*R_cam_FR, R_FR_BR.transpose()*(t_cam_FR-t_FR_BR), lidar_cloud, rgb_cloud);
  #endif
  pcl::toROSMsg(*rgb_cloud, cloudMsg);
  cloudMsg.header.frame_id = "camera_init";
  cloudMsg.header.stamp = ros::Time::now();
  pub_br_cloud.publish(cloudMsg);

  cout<<"push enter"<<endl;
  getchar();
  input_img = cv::imread("/media/sam/CR7/20230613_shenzhen_rosbag/calib_fishcam/fl.png", cv::IMREAD_UNCHANGED);
  #ifdef COLOR_EACH_CAMERA
  color_cloud(fl_intrinsic, fl_coef, input_img, R_cam_FL, t_cam_FL, lidar_cloud, rgb_cloud);
  #else
  color_cloud(fl_intrinsic, fl_coef, input_img, R_FR_FL.transpose()*R_cam_FR, R_FR_FL.transpose()*(t_cam_FR-t_FR_FL), lidar_cloud, rgb_cloud);
  #endif
  pcl::toROSMsg(*rgb_cloud, cloudMsg);
  cloudMsg.header.frame_id = "camera_init";
  cloudMsg.header.stamp = ros::Time::now();
  pub_fl_cloud.publish(cloudMsg);

  ros::Rate loop_rate(1);
  while(ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }
}