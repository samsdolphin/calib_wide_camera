#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

string bag_path, write_path, topic_name;

void bag2img()
{
  rosbag::Bag bag;
  try
  {
    bag.open(bag_path, rosbag::bagmode::Read);
  }
  catch(rosbag::BagException e)
  {
    ROS_ERROR_STREAM("LOADING BAG FAILED: " << e.what());
    return;
  }
  
  std::vector<string> img_topic;
  img_topic.push_back(topic_name);
  rosbag::View img_view(bag, rosbag::TopicQuery(img_topic));
  int cnt = 0, img_cnt = 0;
  cv::Mat rgb_img;
  for(const rosbag::MessageInstance &m: img_view)
  {
    cnt++;
    // if(cnt < 90 || cnt > 1060) continue;
    sensor_msgs::Image image;
    image = *(m.instantiate<sensor_msgs::Image>()); // message type
    cv_bridge::CvImagePtr img_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
    img_ptr->image.copyTo(rgb_img);
    cv::imwrite(write_path + to_string(img_cnt) + ".png", rgb_img);
    img_cnt++;
  }
  cout<<"complete"<<endl;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "bag2img");
  ros::NodeHandle nh("~");

  nh.getParam("bag_path", bag_path);
  nh.getParam("write_path", write_path);
  nh.getParam("topic_name", topic_name);
  
  bag2img();
  // cv::Mat image = cv::imread("/media/sam/CR7/huawei/intrinsic/fr/test/101.png");
  // cv::Mat imageUndistorted;
  // Mat intrinsic = (Mat_<double>(3,3) << 1413.164508112345, 0, 1249.814394660688,
  //                                       0, 1413.629431574755, 1039.340652218145,
  //                                       0, 0, 1);
  // Mat distCoeffs = (Mat_<double>(1,4) << -0.00223105, -0.00146834, -0.00522495, 0.000829206);
  // cv::Size size = { image.cols, image.rows };

  // cv::Mat map1;
  // cv::Mat map2;
  // //cv::fisheye::undistortImage(input_frame,output_frame,cameraMatrix,distortionCoeffs, E, cv::Size(input_frame.cols,input_frame.rows));
  // cv::Mat E = cv::Mat::eye(3, 3, cv::DataType<double>::type);
  // cv::fisheye::initUndistortRectifyMap(intrinsic, distCoeffs, E, intrinsic, size, CV_16SC2, map1, map2);
  // cv::remap(image, imageUndistorted, map1, map2, cv::INTER_LINEAR, CV_HAL_BORDER_CONSTANT);
  // // fisheye::undistortImage(image, imageUndistorted, intrinsic, distCoeffs);
  // cv::imwrite("/media/sam/CR7/huawei/intrinsic/fr/test/101_undist.png", imageUndistorted);

  // imshow("win1", image);
  // imshow("win2", imageUndistorted);
  // waitKey(0);
	// destroyWindow("win1");
  // destroyWindow("win2");

  ros::Rate loop_rate(1);
  while(ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }
}