#include <string>
#include <stdio.h>
#include <fstream>
#include <iostream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "ros/ros.h"
#include <math.h>
#include <rosbag/bag.h>
#include <ceres/ceres.h>

#include "hierarchical_ba.hpp"
#include "tools.hpp"
#include "mypcl.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "visualize");
  ros::NodeHandle nh("~");

  ros::Publisher pub_map = nh.advertise<sensor_msgs::PointCloud2>("/cloud_map", 100);
  ros::Publisher pub_debug = nh.advertise<sensor_msgs::PointCloud2>("/cloud_debug", 100);
  ros::Publisher pub_pose = nh.advertise<geometry_msgs::PoseArray>("/poseArrayTopic", 10);
  ros::Publisher pub_trajectory = nh.advertise<visualization_msgs::Marker>("/trajectory_marker", 100);
  ros::Publisher pub_pose_number = nh.advertise<visualization_msgs::MarkerArray>("/pose_number", 100);

  string file_path;
  bool view_gt, view_ours, kitti2pose, pose2evo, view_cloud, write_pc, write_bag, view_orig;
  double downsample_size, marker_size;
  int pcd_name_fill_num;

  nh.getParam("file_path", file_path);
  nh.getParam("view_gt", view_gt);
  nh.getParam("view_ours", view_ours);
  nh.getParam("view_orig", view_orig);
  nh.getParam("kitti2pose", kitti2pose);
  nh.getParam("pose2evo", pose2evo);
  nh.getParam("downsample_size", downsample_size);
  nh.getParam("view_cloud", view_cloud);
  nh.getParam("write_pc", write_pc);
  nh.getParam("write_bag", write_bag);
  nh.getParam("pcd_name_fill_num", pcd_name_fill_num);
  nh.getParam("marker_size", marker_size);

  sensor_msgs::PointCloud2 debugMsg, cloudMsg, outMsg;
  vector<mypcl::pose> pose_vec;
  vector<double> lidar_times;
  vector_vec3d velocitys, bgs, bas, gravitys;
  rosbag::Bag bag;

  if(write_bag)
    bag.open(file_path + "mybag.bag", rosbag::bagmode::Write);

  if(view_ours)
    // mypcl::loadLiDARState(file_path+"state.txt", lidar_times, pose_vec, velocitys, bgs, bas, gravitys);//即不去掉第一帧
    pose_vec = mypcl::read_pose(file_path + "pose.json");
  // lidar_times = mypcl::readTime(file_path + "time.json");
  size_t pose_size = pose_vec.size();
  cout<<"pose size "<<pose_size<<endl;

  if(kitti2pose) mypcl::write_pose(pose_vec, file_path);
  // if(pose2evo) mypcl::writeEVOPose(lidar_times, pose_vec, file_path);
  if(pose2evo) mypcl::writeKittiEvo(pose_vec, file_path);

  pcl::PointCloud<PointType>::Ptr pc_surf(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr pc_full(new pcl::PointCloud<PointType>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_full(new pcl::PointCloud<pcl::PointXYZRGB>);

  ros::Time cur_t;
  geometry_msgs::PoseArray parray;
  parray.header.frame_id = "camera_init";
  parray.header.stamp = cur_t;
  visualization_msgs::MarkerArray markerArray;

  cout<<"push enter to view"<<endl;
  getchar();
  if(view_cloud)
    for(size_t i = 0; i < pose_size; i++)
    {
      if(view_orig)
        mypcl::loadPCD(file_path, pcd_name_fill_num, pc_surf, i, "undistort/");
      else
        mypcl::loadPCD(file_path, pcd_name_fill_num, pc_surf, i);

      pcl::PointCloud<PointType>::Ptr pc_filtered(new pcl::PointCloud<PointType>);
      pc_filtered->resize(pc_surf->points.size());
      int cnt = 0;
      for(size_t j = 0; j < pc_surf->points.size(); j++)
      {
        // if(pc_surf->points[j].z < 0.3 && pc_surf->points[j].z > -0.3 &&
        //    pc_surf->points[j].x < 0.0 && pc_surf->points[j].x > -0.6 &&
        //    pc_surf->points[j].y < 0.3 && pc_surf->points[j].y > -0.3) continue;
        // Vector3d pt(pc_surf->points[j].x, pc_surf->points[j].y, pc_surf->points[j].z);
        // if(pt.norm() < 1) continue;
        pc_filtered->points[cnt] = pc_surf->points[j];
        cnt++;
      }
      pc_filtered->resize(cnt);
      // pcl::io::savePCDFileBinary(file_path + "pcd/" + to_string(i) + ".pcd", *pc_filtered);
      
      if(write_bag)
      {
        pcl::toROSMsg(*pc_surf, outMsg);
        outMsg.header.frame_id = "PandarXT-32";
        outMsg.header.stamp = ros::Time().fromSec(lidar_times[i]);
        bag.write("/hesai/pandar", outMsg.header.stamp, outMsg);
      }
      
      mypcl::transform_pointcloud(*pc_filtered, *pc_filtered, pose_vec[i].t, pose_vec[i].q);
      if(write_pc) pc_full = mypcl::append_cloud(pc_full, *pc_filtered);
      downsample_voxel(*pc_filtered, downsample_size);

      pcl::toROSMsg(*pc_filtered, cloudMsg);
      cloudMsg.header.frame_id = "camera_init";
      cloudMsg.header.stamp = cur_t;
      pub_map.publish(cloudMsg);

      geometry_msgs::Pose apose;
      apose.orientation.w = pose_vec[i].q.w();
      apose.orientation.x = pose_vec[i].q.x();
      apose.orientation.y = pose_vec[i].q.y();
      apose.orientation.z = pose_vec[i].q.z();
      apose.position.x = pose_vec[i].t(0);
      apose.position.y = pose_vec[i].t(1);
      apose.position.z = pose_vec[i].t(2);
      parray.poses.push_back(apose);
      pub_pose.publish(parray);

      // static tf::TransformBroadcaster br;
      // tf::Transform transform;
      // transform.setOrigin(tf::Vector3(pose_vec[i].t(0), pose_vec[i].t(1), pose_vec[i].t(2)));
      // tf::Quaternion q(pose_vec[i].q.x(), pose_vec[i].q.y(), pose_vec[i].q.z(), pose_vec[i].q.w());
      // transform.setRotation(q);
      // br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "camera_init", "turtle_name"));

      // publish pose trajectory
      visualization_msgs::Marker marker;
      marker.header.frame_id = "camera_init";
      marker.header.stamp = cur_t;
      marker.ns = "basic_shapes";
      marker.id = i;
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.pose.position.x = pose_vec[i].t(0);
      marker.pose.position.y = pose_vec[i].t(1);
      marker.pose.position.z = pose_vec[i].t(2);
      pose_vec[i].q.normalize();
      marker.pose.orientation.x = pose_vec[i].q.x();
      marker.pose.orientation.y = pose_vec[i].q.y();
      marker.pose.orientation.z = pose_vec[i].q.x();
      marker.pose.orientation.w = pose_vec[i].q.w();
      marker.scale.x = marker_size; // Set the scale of the marker -- 1x1x1 here means 1m on a side
      marker.scale.y = marker_size;
      marker.scale.z = marker_size;
      marker.color.r = float(1-float(i)/pose_size);
      marker.color.g = float(float(i)/pose_size);
      marker.color.b = float(float(i)/pose_size);
      marker.color.a = 1.0;
      marker.lifetime = ros::Duration();
      pub_trajectory.publish(marker);

      // publish pose number
      visualization_msgs::Marker marker_txt;
      marker_txt.header.frame_id = "camera_init";
      marker_txt.header.stamp = cur_t;
      marker_txt.ns = "marker_txt";
      marker_txt.id = i; // Any marker sent with the same namespace and id will overwrite the old one
      marker_txt.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      ostringstream str;
      str << i;
      marker_txt.text = str.str();
      marker.action = visualization_msgs::Marker::ADD;
      marker_txt.action = visualization_msgs::Marker::ADD;
      marker_txt.pose.position.x = pose_vec[i].t(0)+marker_size;
      marker_txt.pose.position.y = pose_vec[i].t(1)+marker_size;
      marker_txt.pose.position.z = pose_vec[i].t(2);
      marker_txt.pose.orientation.x = pose_vec[i].q.x();
      marker_txt.pose.orientation.y = pose_vec[i].q.y();
      marker_txt.pose.orientation.z = pose_vec[i].q.x();
      marker_txt.pose.orientation.w = 1.0;
      marker_txt.scale.x = marker_size;
      marker_txt.scale.y = marker_size;
      marker_txt.scale.z = marker_size;
      marker_txt.color.r = 1.0f;
      marker_txt.color.g = 1.0f;
      marker_txt.color.b = 1.0f;
      marker_txt.color.a = 1.0;
      marker_txt.lifetime = ros::Duration();
      if(i%GAP == 0) markerArray.markers.push_back(marker_txt);
      pub_pose_number.publish(markerArray);

      ros::Duration(0.001).sleep();
    }
  bag.close();

  if(write_pc)
  {
    // cout<<"working on statistical filter"<<endl;
    // pcl::StatisticalOutlierRemoval<PointType> sor;
    // sor.setInputCloud(pc_full);//设置待滤波的点云
    // sor.setMeanK(10);//设置在进行统计时考虑查询点邻居点数
    // sor.setStddevMulThresh(1.0);//设置判断是否为离群点的阈值
    // sor.filter(*pc_full);//将滤波结果保存在cloud_filtered中

    // cout<<"working on pass through filter"<<endl;
    // pcl::PassThrough<PointType> pass;
    // pass.setInputCloud(pc_full);
    // pass.setFilterFieldName("z");
    // pass.setFilterLimits(0.1, 1.5);
    // pass.filter(*pc_full);

    cout<<"working on downsampling"<<endl;
    downsample_voxel(*pc_full, 0.01);

    pcl::io::savePCDFileBinary(file_path + "full.pcd", *pc_full);
    cout<<"pointcloud saved"<<endl;

    int part = 0;
    int total_part = pc_full->points.size()*1e-6;
    cout<<"total part "<<total_part<<endl;
    if(part < total_part)
    {
      cout<<"publish part "<<part<<endl;
      pcl::PointCloud<PointType>::Ptr pc_(new pcl::PointCloud<PointType>);
      pc_->resize(1e6);
      int cnt = 0;
      for(size_t j = part*1e6; j < (part+1)*1e6; j++)
      {
        pc_->points[cnt].x = pc_full->points[j].x;
        pc_->points[cnt].y = pc_full->points[j].y;
        pc_->points[cnt].z = pc_full->points[j].z;
        cnt++;
      }
      part++;

      pcl::toROSMsg(*pc_, debugMsg);
      debugMsg.header.frame_id = "camera_init";
      debugMsg.header.stamp = cur_t;
      pub_map.publish(debugMsg);
    }
  }

  ros::Rate loop_rate(1);
  while(ros::ok())
  {
    // cout<<"push enter to view cloud clockwise"<<endl;
    // getchar();
    // // Quaterniond dq(0.9999619, 0, 0, 0.0087265), q0(1, 0, 0, 0); // 1 degree
    // Quaterniond dq(0.9996573, 0, 0, 0.0261769), q0(1, 0, 0, 0); // 3 degree
    // for(int i = 0; i < 120; i++)
    // {
    //   static tf::TransformBroadcaster br;
    //   tf::Transform transform;
    //   transform.setOrigin(tf::Vector3(0, 0, 0));
    //   q0*=dq;
    //   tf::Quaternion q(q0.x(), q0.y(), q0.z(), q0.w());
    //   transform.setRotation(q);
    //   br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "camera_init", "turtle_name"));
    //   ros::spinOnce();
    //   loop_rate.sleep();
    // }
    ros::spinOnce();
    loop_rate.sleep();
  }
}