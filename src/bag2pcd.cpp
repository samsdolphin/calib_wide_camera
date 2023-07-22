#include <string>
#include <fstream>
#include <iostream>

#include <stdio.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <tf/transform_broadcaster.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <ceres/ceres.h>
#include "ros/ros.h"

// #include "tools.hpp"
#include "mypcl.hpp"

// typedef pcl::PointXYZI PointType;

using namespace std;
using namespace Eigen;

bool exit_flag = false;
int pc_cnt = 0, msg_cnt = 0;
string in_datapath = "/media/sam/LiT7/data/fba/new_device/calib/ex_1/";

pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);

void PressEnterToExit(void)
{
  int c;
  while((c = getchar()) != '\n' && c != EOF) ;
  fprintf(stderr, "\nPress enter to exit.\n");
  while(getchar() != '\n') ;
  exit_flag = true;
  sleep(1);
}

void lidar_callback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  msg_cnt++;
  // if(msg_cnt > 100)
  {
    pcl::PointCloud<PointType> pc;
    pcl::fromROSMsg(*msg, pc);
    std::stringstream ss;
    ss << setw(5) << setfill('0') << pc_cnt;
    // for (size_t i = 0; i < pc.size(); i++)
    // {
    //   PointType pi = pc.points[i];
    //   double range = sqrt(pi.x * pi.x + pi.y * pi.y + pi.z * pi.z);
    //   // 0.225 good
    //   double calib_vertical_angle = DEG2RAD(0.2);
    //   // Eigen::Vector3d euler_angle(0, deg2rad(-0.6), 0); 
    //   // Eigen::Matrix3d calib_rotation; 
    //   // calib_rotation = 
    //   // Eigen::AngleAxisd(euler_angle[2], Eigen::Vector3d::UnitZ()) * 
    //   // Eigen::AngleAxisd(euler_angle[1], Eigen::Vector3d::UnitY()) * 
    //   // Eigen::AngleAxisd(euler_angle[0], Eigen::Vector3d::UnitX()); 
    //   // Eigen::Vector3d pv(pi.x, pi.y, pi.z); 
    //   // pv = calib_rotation * pv; 
    //   // pi.x = pv[0]; 
    //   // pi.y = pv[1]; 
    //   // pi.z = pv[2]; 
    //   double vertical_angle = atan2(pi.z, range) + calib_vertical_angle;
    //   double horizon_angle = atan2(pi.y, pi.x);
    //   pi.z = range * tan(vertical_angle);
    //   double project_len = range * cos(vertical_angle);
    //   pi.x = project_len * cos(horizon_angle);
    //   pi.y = project_len * sin(horizon_angle);
    //   pc.points[i] = pi;
    // }
    pcl::io::savePCDFileBinary(in_datapath + ss.str() + ".pcd", pc);
    pc_cnt++;
  }
}

void avia_callback(const livox_ros_driver::CustomMsg::ConstPtr& msg)
{
  pcl::PointCloud<PointType>::Ptr _pc(new pcl::PointCloud<PointType>);
  size_t pc_size = msg->point_num;
  _pc->points.resize(pc_size);
  for(size_t i = 0; i < pc_size; i++)
  {
    _pc->points[i].x = msg->points[i].x;
    _pc->points[i].y = msg->points[i].y;
    _pc->points[i].z = msg->points[i].z;
  }
  pc = mypcl::append_cloud(pc, *_pc);
  msg_cnt++;
  ROS_INFO_STREAM("pc size "<<pc->points.size()<<" "<<msg_cnt);
}

void hesai_callback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  msg_cnt++;
  // if(msg_cnt > 100)
  {
    pcl::PointCloud<pcl::PointXYZI> pc_;
    pcl::fromROSMsg(*msg, pc_);
    pc = mypcl::append_cloud(pc, pc_);
    cout<<"pc size "<<pc->points.size()<<" "<<msg_cnt<<endl;
    // std::stringstream ss;
    // ss << setw(0) << setfill('0') << pc_cnt;
    // pcl::io::savePCDFileBinary(in_datapath + ss.str() + ".pcd", pc);
    pc_cnt++;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "rosbag2pcd");
  ros::NodeHandle nh("~");

  // ros::Subscriber sub_points = nh.subscribe("/velodyne_points", 1000, lidar_callback);
  // ros::Subscriber sub_points = nh.subscribe("/os1_cloud_node/points", 1000, lidar_callback);
  ros::Subscriber sub_points = nh.subscribe("/livox/lidar", 1000, avia_callback);
  // ros::Subscriber sub_points = nh.subscribe("//hesai/pandar_points", 1000, hesai_callback);

  // pcl::io::savePCDFileBinary(in_datapath + "0.pcd", *pc);

  ros::Rate loop_rate(100);
  while(ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
    if(msg_cnt == 47)
    {
      pcl::io::savePCDFileBinary(in_datapath + "0.pcd", *pc);
      break;
    }
  }
  return 0;
}