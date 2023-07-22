#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// #define time_lag 732791263.067978 // from zhufangcheng
#define time_lag 733720207.33210661

ros::Publisher pubIMU_merged;
sensor_msgs::Imu IMU_merged;

// int msg_cnt = 0;
void acc_gyro_callback(const geometry_msgs::Vector3Stamped::ConstPtr& acc, const geometry_msgs::Vector3Stamped::ConstPtr& gyro)
{
  IMU_merged.header = acc->header;
  IMU_merged.header.stamp.sec -= time_lag;
  IMU_merged.angular_velocity.x = gyro->vector.x;
  IMU_merged.angular_velocity.y = gyro->vector.y;
  IMU_merged.angular_velocity.z = gyro->vector.z;
  IMU_merged.linear_acceleration.x = acc->vector.x;
  IMU_merged.linear_acceleration.y = acc->vector.y;
  IMU_merged.linear_acceleration.z = acc->vector.z;
  pubIMU_merged.publish(IMU_merged);
  // printf("msg_cnt = %d\n", msg_cnt++);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "combine_imu");
  ros::NodeHandle nh;

  pubIMU_merged = nh.advertise<sensor_msgs::Imu>("/hesai/pandar_imu", 1000);

  message_filters::Subscriber<geometry_msgs::Vector3Stamped> acc_sub(nh, "/imu/acceleration", 200);
  message_filters::Subscriber<geometry_msgs::Vector3Stamped> gyro_sub(nh, "/imu/angular_velocity", 200);
  typedef message_filters::sync_policies::ApproximateTime<geometry_msgs::Vector3Stamped, geometry_msgs::Vector3Stamped> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(200), acc_sub, gyro_sub);
  sync.registerCallback(boost::bind(&acc_gyro_callback, _1, _2));

  ros::Rate loop_rate(200);
  while(ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}