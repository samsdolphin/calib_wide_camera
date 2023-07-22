#include <iostream>
#include <fstream>
#include <string>

#include <mutex>
#include <assert.h>
#include <ros/ros.h>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <tf/transform_broadcaster.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include "hierarchical_ba.hpp"
#include "tools.hpp"
#include "mypcl.hpp"

using namespace std;
using namespace Eigen;

typedef Matrix<double, 6, 6> Matrix6d;

double residual_cur, residual_pre;
double dsp_sz1, voxel_size, rej_ratio1, rej_ratio2;
string in_datapath;
std::ofstream poseFile, aggFile, hessFile, newPoseFile;
int max_iter, tail, gap_num, last_win_size, pcd_fill_num;
int THR_NUM;

void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
               pcl::PointCloud<PointType>& pl_feat,
							 Eigen::Quaterniond q, Eigen::Vector3d t, int fnum,
               double voxel_size, int window_size, float eigen_ratio)
{
	float loc_xyz[3];
	for(PointType& p_c: pl_feat.points)
	{
		Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
		Eigen::Vector3d pvec_tran = q * pvec_orig + t;

		for(int j = 0; j < 3; j++)
		{
			loc_xyz[j] = pvec_tran[j] / voxel_size;
			if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
		}

		VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
		auto iter = feat_map.find(position);
		if(iter != feat_map.end())
		{
      iter->second->vec_orig[fnum].push_back(pvec_orig);
      iter->second->vec_tran[fnum].push_back(pvec_tran);
			
      iter->second->sig_orig[fnum].push(pvec_orig);
      iter->second->sig_tran[fnum].push(pvec_tran);
		}
		else
		{
			OCTO_TREE_ROOT* ot = new OCTO_TREE_ROOT(window_size, eigen_ratio);
			ot->vec_orig[fnum].push_back(pvec_orig);
			ot->vec_tran[fnum].push_back(pvec_tran);
			ot->sig_orig[fnum].push(pvec_orig);
			ot->sig_tran[fnum].push(pvec_tran);

			ot->voxel_center[0] = (0.5+position.x) * voxel_size;
			ot->voxel_center[1] = (0.5+position.y) * voxel_size;
			ot->voxel_center[2] = (0.5+position.z) * voxel_size;
			ot->quater_length = voxel_size / 4.0;
			ot->layer = 0;
			feat_map[position] = ot;
		}
	}
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "hierarchical_bundle_adjustment");
	ros::NodeHandle nh("~");

	ros::Publisher pub_map = nh.advertise<sensor_msgs::PointCloud2>("/map_surf", 100);
	ros::Publisher pub_debug = nh.advertise<sensor_msgs::PointCloud2>("/debug_surf", 100);
	ros::Publisher pub_residual = nh.advertise<sensor_msgs::PointCloud2>("/residual", 100);
	ros::Publisher pub_pose = nh.advertise<geometry_msgs::PoseArray>("/poseArrayTopic", 100);

  double dsp_sz2;
  string out_datapath;
  float eigen_ratio2;

	nh.getParam("in_datapath", in_datapath);
	nh.getParam("out_datapath", out_datapath);
	nh.getParam("max_iter", max_iter);
	nh.getParam("voxel_size", voxel_size);
	nh.getParam("dsp_sz1", dsp_sz1);
	nh.getParam("dsp_sz2", dsp_sz2);
	nh.getParam("rej_ratio1", rej_ratio1);
	nh.getParam("rej_ratio2", rej_ratio2);
	// nh.getParam("eigen_ratio1", eigen_ratio1);
	nh.getParam("eigen_ratio2", eigen_ratio2);
	nh.getParam("THR_NUM", THR_NUM);
	nh.getParam("pcd_fill_num", pcd_fill_num);

	sensor_msgs::PointCloud2 dbgMsg, mapMsg;
	vector<mypcl::pose> pose_vec;
	size_t pose_size;
	
	ros::Time t_begin, t_end, cur_t = ros::Time::now();
	double avg_time = 0.0, avg_err = 0.0;
  std::ofstream tErrFile;

  vector<pcl::PointCloud<PointType>::Ptr> src_pc;
  // pose_vec.clear(); avg_time = 0;
  pose_vec = mypcl::read_pose(in_datapath + "pose.json");
  int window_size = pose_vec.size();
  src_pc.resize(window_size);
  std::cout<<"window_size "<<window_size<<std::endl;
  // hessFile.open(in_datapath + "process1/hessian.json", std::ofstream::trunc);
  // hessFile.close();

  vector<mypcl::pose> pose_vec_ori = mypcl::read_pose(in_datapath + "pose.json");
  size_t pose_vec_ori_size = pose_vec_ori.size();
  int tail = (pose_vec_ori_size - WIN_SIZE) % GAP;
  int gap_num = (pose_vec_ori_size -WIN_SIZE) / GAP;
  int last_win_size = pose_vec_ori_size - GAP * (gap_num+1);
  residual_cur = 0; residual_pre = 0;
  double pose_err = 0;

  // vector<mypcl::pose> pose_aggregate = mypcl::read_pose(in_datapath + "aggregate.json");

  cur_t = ros::Time::now();
  for(int i = 0; i < window_size; i++)
  {
    pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
    mypcl::loadPCD(in_datapath, pcd_fill_num, pc, i);
    src_pc[i] = pc;
  }
  cout<<"load pcd time "<<ros::Time::now().toSec()-cur_t.toSec()<<endl;

  size_t max_mem = 0;
  size_t mem_cost = 0;
  for(int loop = 0; loop < max_iter; loop++)
  {
    std::cout<<"---------------------"<<std::endl;
    std::cout<<"iteration "<<loop<<std::endl;
    t_begin = ros::Time::now();
    unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
    vector<IMUST> x_buf(window_size);
    
    double t2 = ros::Time::now().toSec();
    for(int i = 0; i < window_size; i++)
    {
      if(dsp_sz2 > 0) downsample_voxel(*src_pc[i], dsp_sz2);
      cut_voxel(surf_map, *src_pc[i], pose_vec[i].q, pose_vec[i].t, i,
                voxel_size, window_size, eigen_ratio2);
    }
    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->recut();

    for(int i = 0; i < window_size; i++)
    {
      x_buf[i].R = pose_vec[i].q.toRotationMatrix();
      x_buf[i].p = pose_vec[i].t;
    }

    VOX_HESS voxhess(window_size);
    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->tras_opt(voxhess);
    cout<<"cut voxel time "<<ros::Time::now().toSec()-t2<<endl;
    
    double t3 = ros::Time::now().toSec();
    VOX_OPTIMIZER opt_lsv(window_size);
    opt_lsv.remove_outlier(x_buf, voxhess, rej_ratio2);
    Eigen::MatrixXd Hess_cur(6*window_size, 6*window_size);
    opt_lsv.damping_iter(x_buf, voxhess, residual_cur, Hess_cur, mem_cost);
    cout<<"solve optimization time "<<ros::Time::now().toSec()-t3<<endl;

    for(int i = 0; i < window_size; i++)
      assign_qt(pose_vec[i].q, pose_vec[i].t, Quaterniond(x_buf[i].R), x_buf[i].p);

    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      delete iter->second;
    
    t_end = ros::Time::now();
    std::cout<<"time cost "<<(t_end - t_begin).toSec()<<std::endl;
    avg_time += (t_end - t_begin).toSec();

    cout<<"residual absolute "<<abs(residual_pre-residual_cur)<<", "
      <<"percentage "<<abs(residual_pre-residual_cur)/abs(residual_cur)<<endl;
    
    if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == max_iter-1)
    {
      // if(max_mem < mem_cost) max_mem = mem_cost;
      // hessFile.open(in_datapath + "process1/hessian.json", std::ofstream::app);
      // #ifdef FULL_HESS
      // for(int i = 0; i < window_size-1; i++)
      //   for(int j = i+1; j < window_size; j++)
      //   {
      //     Matrix6d hess = Hess_cur.block(6*i, 6*j, 6, 6);
      //     for(int row = 0; row < 6; row++)
      //       for(int col = 0; col < 6; col++)
      //         hessFile << hess(row, col) << ((row*col==25)?"":" ");
      //     if(i < window_size-2) hessFile << "\n";
      //   }
      // #else
      // for(int i = 0; i < window_size-1; i++)
      // {
      //   Matrix6d hess = Hess_cur.block(6*i, 6*i+6, 6, 6);
      //   for(int row = 0; row < 6; row++)
      //     for(int col = 0; col < 6; col++)
      //       hessFile << hess(row, col) << ((row*col==25)?"":" ");
      //   if(i < window_size-2) hessFile << "\n";
      // }
      // #endif
      // hessFile.close();
      break;
    }
    residual_pre = residual_cur;
  }
  cout<<"max mem "<<double(max_mem/1048576.0)<<endl;
  std::cout<<"---------------------"<<std::endl;
  std::cout<<"pyramid 2 total time "<<avg_time<<std::endl;
  std::cout<<"maximum memory cost "<<max_mem/1048576.0<<std::endl;
  mypcl::write_pose(pose_vec, in_datapath);

  cout<<"complete"<<endl;
}