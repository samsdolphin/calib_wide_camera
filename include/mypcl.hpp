#ifndef MYPCL_HPP
#define MYPCL_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>

typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vector_vec3d;
typedef std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > vector_quad;
// typedef pcl::PointXYZINormal PointType;
// typedef pcl::PointXYZ PointType;
typedef pcl::PointXYZI PointType;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

namespace mypcl
{
  struct pose
  {
    pose(Eigen::Quaterniond _q = Eigen::Quaterniond(1, 0, 0, 0),
         Eigen::Vector3d _t = Eigen::Vector3d(0, 0, 0)):q(_q), t(_t){}
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
  };

  void loadLiDARState(std::string filePath, std::vector<double>& lidar_time, std::vector<pose>& pose_vec,
                      vector_vec3d& velocitys, vector_vec3d& bgs, vector_vec3d& bas, vector_vec3d& gravitys,
                      bool skip_last = false)
  {
    std::fstream file;
    file.open(filePath);
    double t, tx, ty, tz, qw, qx, qy, qz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz, gx, gy, gz;
    int cnt = 0;
    while(!file.eof())
    {
      file >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw
           >> vx >> vy >> vz >> bgx >> bgy >> bgz >> bax >> bay >> baz
           >> gx >> gy >> gz;
      lidar_time.push_back(t);
      Eigen::Quaterniond q(qw, qx, qy, qz);
      Eigen::Vector3d tran(tx, ty, tz);
      if(cnt == 0 && skip_last)//第一个位姿的R,p,v,bg,ba重复添加一次
      {
        pose_vec.push_back(pose(q, tran));
        velocitys.push_back(Eigen::Vector3d(vx, vy, vz));
        bgs.push_back(Eigen::Vector3d(bgx, bgy, bgz));
        bas.push_back(Eigen::Vector3d(bax, bay, baz));
        gravitys.push_back(Eigen::Vector3d(gx, gy, gz));
      }
      pose_vec.push_back(pose(q, tran));
      velocitys.push_back(Eigen::Vector3d(vx, vy, vz));
      bgs.push_back(Eigen::Vector3d(bgx, bgy, bgz));
      bas.push_back(Eigen::Vector3d(bax, bay, baz));
      gravitys.push_back(Eigen::Vector3d(gx, gy, gz));
      cnt++;
    }
    file.close();
    if(skip_last)
    {
      pose_vec.pop_back();
      velocitys.pop_back();
      bgs.pop_back();
      bas.pop_back();
      gravitys.pop_back();
    }
  }

  std::vector<pose> loadLiDARState(std::string filePath, bool skip_last = false)
  {
    std::vector<pose> pose_vec;
    std::fstream file;
    file.open(filePath);
    double t, tx, ty, tz, qw, qx, qy, qz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz, gx, gy, gz;
    while(!file.eof())
    {
      file >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw
           >> vx >> vy >> vz >> bgx >> bgy >> bgz >> bax >> bay >> baz
           >> gx >> gy >> gz;
      Eigen::Quaterniond q(qw, qx, qy, qz);
      Eigen::Vector3d tran(tx, ty, tz);
      if(skip_last)
        pose_vec.push_back(pose(q, tran));
      pose_vec.push_back(pose(q, tran));
    }
    file.close();
    if(skip_last) pose_vec.pop_back();
    return pose_vec;
  }

  void writeLiDARState(std::ofstream& file, double lidar_time, pose pose_vec, Eigen::Vector3d velocity,
                       Eigen::Vector3d bg, Eigen::Vector3d ba, Eigen::Vector3d gravity, bool next_line = true)
  {
    file << std::setprecision(16) << lidar_time << " " << std::setprecision(6)
         << pose_vec.t(0) << " " << pose_vec.t(1) << " " << pose_vec.t(2) << " "
         << pose_vec.q.x() << " " << pose_vec.q.y() << " " << pose_vec.q.z() << " " << pose_vec.q.w() << " "
         << velocity(0) << " " << velocity(1) << " " << velocity(2) << " "
         << bg(0) << " " << bg(1) << " " << bg(2) << " " << ba(0) << " " << ba(1) << " " << ba(2) << " "
         << gravity(0) << " " << gravity(1) << " " << gravity(2);
    if(next_line) file << std::endl;
  }

  void logPose(std::ofstream& file, pose pose_vec, bool next_line = true)
  {
    file << std::setprecision(6)
         << pose_vec.t(0) << " " << pose_vec.t(1) << " " << pose_vec.t(2) << " "
         << pose_vec.q.w() << " " << pose_vec.q.x() << " " << pose_vec.q.y() << " " << pose_vec.q.z();
    if(next_line) file << std::endl;
  }

  void loadPCD(std::string filePath, int pcd_fill_num, pcl::PointCloud<PointType>::Ptr& pc, int num,
               std::string prefix = "")
  {
    std::stringstream ss;
    if(pcd_fill_num > 0)
      ss << std::setw(pcd_fill_num) << std::setfill('0') << num;
    else
      ss << num;
    pcl::io::loadPCDFile(filePath + prefix + ss.str() + ".pcd", *pc);
  }

  void savdPCD(std::string filePath, int pcd_fill_num, pcl::PointCloud<PointType>::Ptr& pc, int num)
  {
    std::stringstream ss;
    if(pcd_fill_num > 0)
      ss << std::setw(pcd_fill_num) << std::setfill('0') << num;
    else
      ss << num;
    pcl::io::savePCDFileBinary(filePath + ss.str() + ".pcd", *pc);
  }
  
  std::vector<double> readTime(std::string filename)
  {
    std::vector<double> time_vec;
    std::fstream file;
    file.open(filename);
    double t;
    while(!file.eof())
    {
      file >> t;
      time_vec.push_back(t);
    }
    return time_vec;
  }

  std::vector<pose> readEvoPose(std::string filename)
  {
    std::vector<pose> pose_vec;
    std::fstream file;
    file.open(filename);
    size_t cnt = 0;
    double x, y, z, qw, qx, qy, qz, t_;
    while(!file.eof())
    {
      file >> t_ >> x >> y >> z >> qx >> qy >> qz >> qw;
      Eigen::Quaterniond q(qw, qx, qy, qz);
      Eigen::Vector3d t(x, y, z);
      pose_vec.push_back(pose(q, t));
      cnt++;
    }
    return pose_vec;
  }

  std::vector<double> readEvoTime(std::string filename)
  {
    std::vector<double> time_vec;
    std::fstream file;
    file.open(filename);
    size_t cnt = 0;
    double x, y, z, qw, qx, qy, qz, t_;
    while(!file.eof())
    {
      file >> t_ >> x >> y >> z >> qx >> qy >> qz >> qw;
      time_vec.push_back(t_);
      cnt++;
    }
    return time_vec;
  }

  std::vector<Matrix6d> readCovariance(std::string filename)
  {
    std::vector<Matrix6d> cov_vec;
    Matrix6d cov;
    std::fstream file;
    file.open(filename);
    while(!file.eof())
    {
      for(int row = 0; row < 6; row++)
        for(int col = 0; col < 6; col++)
          file >> cov(row, col);
      cov_vec.push_back(cov);
    }
    return cov_vec;
  }

  std::vector<pose> loadXyzrpy(std::string filename)
  {
    std::vector<pose> pose_vec;
    std::fstream file;
    file.open(filename);
    size_t cnt = 0;
    double x, y, z, roll, pitch, yaw;
    Eigen::Matrix3d Rx, Ry, Rz, R;
    while(!file.eof())
    {
      file >> x >> y >> z >> roll >> pitch >> yaw;
      Rx << 1, 0, 0, 0, cos(roll), -sin(roll), 0, sin(roll), cos(roll);
      Ry << cos(pitch), 0, sin(pitch), 0, 1, 0, -sin(pitch), 0, cos(pitch);
      Rz << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1;
      R = Ry * Rx * Rz;
      Eigen::Quaterniond q(R);
      Eigen::Vector3d t(x, y, z);
      pose_vec.push_back(pose(q, t));
      cnt++;
    }
    return pose_vec;
  }

  std::vector<pose> loadKittiPose(std::string filename)
  {
    std::vector<pose> pose_vec;
    std::fstream file;
    file.open(filename);
    size_t cnt = 0;
    double r11, r12, r13, r14, r21, r22, r23, r24, r31, r32, r33, r34;
    while(!file.eof())
    {
      file >> r11 >> r12 >> r13 >> r14 >>
        r21 >> r22 >> r23 >> r24 >> r31 >> r32 >> r33 >> r34;
      Eigen::Matrix3d R;
      R << r11, r12, r13, r21, r22, r23, r31, r32, r33;
      Eigen::Quaterniond q(R);
      Eigen::Vector3d t(r14, r24, r34);
      pose_vec.push_back(pose(q, t));
      cnt++;
    }
    file.close();
    return pose_vec;
  }

  std::vector<pose> read_pose(std::string path)
  {
    std::vector<pose> pose_vec;
    std::fstream file;
    file.open(path);
    double tx, ty, tz, w, x, y, z;
    while(!file.eof())
    {
      file >> tx >> ty >> tz >> w >> x >> y >> z;
      pose_vec.push_back(pose(Eigen::Quaterniond(w, x, y, z), Eigen::Vector3d(tx, ty, tz)));
    }
    file.close();
    return pose_vec;
  }

  void transform_pointcloud(pcl::PointCloud<PointType> const& pc_in,
                            pcl::PointCloud<PointType>& pt_out,
                            Eigen::Vector3d t,
                            Eigen::Quaterniond q)
  {
    size_t size = pc_in.points.size();
    pt_out.points.resize(size);
    for(size_t i = 0; i < size; i++)
    {
      Eigen::Vector3d pt_cur(pc_in.points[i].x, pc_in.points[i].y, pc_in.points[i].z);
      Eigen::Vector3d pt_to;
      // if(pt_cur.norm()<0.3) continue;
      pt_to = q * pt_cur + t;
      pt_out.points[i].x = pt_to.x();
      pt_out.points[i].y = pt_to.y();
      pt_out.points[i].z = pt_to.z();
      // pt_out.points[i].r = pc_in.points[i].r;
      // pt_out.points[i].g = pc_in.points[i].g;
      // pt_out.points[i].b = pc_in.points[i].b;
    }
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr append_cloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc1,
                                                pcl::PointCloud<pcl::PointXYZI> pc2)
  {
    size_t size1 = pc1->points.size();
    size_t size2 = pc2.points.size();
    pc1->points.resize(size1 + size2);
    for(size_t i = size1; i < size1 + size2; i++)
    {
      pc1->points[i].x = pc2.points[i-size1].x;
      pc1->points[i].y = pc2.points[i-size1].y;
      pc1->points[i].z = pc2.points[i-size1].z;
      // pc1->points[i].r = pc2.points[i-size1].r;
      // pc1->points[i].g = pc2.points[i-size1].g;
      // pc1->points[i].b = pc2.points[i-size1].b;
      pc1->points[i].intensity = pc2.points[i-size1].intensity;
    }
    return pc1;
  }

  // pcl::PointCloud<PointType>::Ptr append_cloud(pcl::PointCloud<PointType>::Ptr pc1,
  //                                               pcl::PointCloud<PointType> pc2)
  // {
  //   size_t size1 = pc1->points.size();
  //   size_t size2 = pc2.points.size();
  //   pc1->points.resize(size1 + size2);
  //   for(size_t i = size1; i < size1 + size2; i++)
  //   {
  //     pc1->points[i].x = pc2.points[i-size1].x;
  //     pc1->points[i].y = pc2.points[i-size1].y;
  //     pc1->points[i].z = pc2.points[i-size1].z;
  //     // pc1->points[i].r = pc2.points[i-size1].r;
  //     // pc1->points[i].g = pc2.points[i-size1].g;
  //     // pc1->points[i].b = pc2.points[i-size1].b;
  //     // pc1->points[i].intensity = pc2.points[i-size1].intensity;
  //   }
  //   return pc1;
  // }

  double compute_inlier_ratio(std::vector<double> residuals, double ratio)
  {
    std::set<double> dis_vec;
    for(size_t i = 0; i < (size_t)(residuals.size() / 3); i++)
      dis_vec.insert(fabs(residuals[3 * i + 0]) +
                     fabs(residuals[3 * i + 1]) + fabs(residuals[3 * i + 2]));

    return *(std::next(dis_vec.begin(), (int)((ratio) * dis_vec.size())));
  }

  void write_pose(std::vector<pose>& pose_vec, std::string path, 
                  bool startfrom0 = true, double ratio = 1)
  {
    std::ofstream file;
    file.open(path + "pose.json", std::ofstream::trunc);
    file.close();
    Eigen::Quaterniond q0(pose_vec[0].q.w(), pose_vec[0].q.x(), 
                          pose_vec[0].q.y(), pose_vec[0].q.z());
    Eigen::Vector3d t0(pose_vec[0].t(0), pose_vec[0].t(1), pose_vec[0].t(2));
    file.open(path + "pose.json", std::ofstream::app);
    if(!startfrom0)
    {
      q0 = Eigen::Quaterniond(1, 0, 0, 0);
      t0 << 0, 0, 0;
    }
    for(size_t i = 0; i < pose_vec.size(); i++)
    {
      pose_vec[i].t << q0.inverse()*(pose_vec[i].t-t0);
      pose_vec[i].q.w() = (q0.inverse()*pose_vec[i].q).w();
      pose_vec[i].q.x() = (q0.inverse()*pose_vec[i].q).x();
      pose_vec[i].q.y() = (q0.inverse()*pose_vec[i].q).y();
      pose_vec[i].q.z() = (q0.inverse()*pose_vec[i].q).z();
      file << ratio*pose_vec[i].t(0) << " "
           << ratio*pose_vec[i].t(1) << " "
           << ratio*pose_vec[i].t(2) << " "
           << pose_vec[i].q.w() << " " << pose_vec[i].q.x() << " "
           << pose_vec[i].q.y() << " " << pose_vec[i].q.z();
      if(i < pose_vec.size()-1) file << "\n";
    }
    file.close();
  }

  void writeEVOPose(std::vector<double>& lidar_times, std::vector<pose>& pose_vec, std::string path)
  {
    std::ofstream file;
    file.open(path + "evo_pose.txt", std::ofstream::trunc);
    for(size_t i = 1; i < pose_vec.size(); i++)
    {
      file << std::setprecision(12) << lidar_times[i] << " "
           << pose_vec[i-1].t(0) << " " << pose_vec[i-1].t(1) << " " << pose_vec[i-1].t(2) << " "
           << pose_vec[i-1].q.x() << " " << pose_vec[i-1].q.y() << " "
           << pose_vec[i-1].q.z() << " " << pose_vec[i-1].q.w();
      if(i < pose_vec.size()-1) file << "\n";
    }
    file.close();
  }

  void writeKittiEvo(std::vector<pose>& pose_vec, std::string path)
  {
    std::ofstream file;
    file.open(path + "evo_pose.txt", std::ofstream::trunc);
    for(size_t i = 0; i < pose_vec.size(); i++)
    {
      file << i+1 << " " << std::setprecision(6)
           << pose_vec[i].t(0) << " " << pose_vec[i].t(1) << " " << pose_vec[i].t(2) << " "
           << pose_vec[i].q.x() << " " << pose_vec[i].q.y() << " "
           << pose_vec[i].q.z() << " " << pose_vec[i].q.w();
      if(i < pose_vec.size()-1) file << "\n";
    }
    file.close();
  }

  // void make_arrow_func(pcl::PointCloud<PointType>& pl, visualization_msgs::MarkerArray& marker_array,
  //                      int& mid, ros::Time ct)
  // {
  //   for(uint i=0; i<pl.size(); i++)
  //   {
  //     visualization_msgs::Marker marker;
  //     marker.header.frame_id = "camera_init";
  //     marker.header.stamp = ct;
  //     marker.id = mid;
  //     mid++;
  //     marker.action = visualization_msgs::Marker::ADD;
  //     marker.type = visualization_msgs::Marker::ARROW;
  //     marker.color.a = 1;
  //     marker.color.r = 1;
  //     marker.color.g = 0;
  //     marker.color.b = 0;
  //     marker.scale.x = 0.02;
  //     marker.scale.y = 0.05;
  //     marker.scale.z = 0.1;
  //     marker.lifetime = ros::Duration();
  //     geometry_msgs::Point apoint;
  //     apoint.x = pl[i].x;
  //     apoint.y = pl[i].y;
  //     apoint.z = pl[i].z;
  //     marker.points.push_back(apoint);
  //     apoint.x += pl[i].normal_x;
  //     apoint.y += pl[i].normal_y;
  //     apoint.z += pl[i].normal_z;
  //     marker.points.push_back(apoint);
  //     marker_array.markers.push_back(marker);
  //   }
  // }
}

#endif