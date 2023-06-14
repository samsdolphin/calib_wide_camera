cmake_minimum_required(VERSION 3.0.2)
project(calib_wide_camera)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(PCL REQUIRED)
find_package(OpenCV)
find_package(Threads)
find_package(Ceres REQUIRED)
find_package(GTSAM REQUIRED QUIET)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

catkin_package(
 CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

add_executable(bag2img src/bag2img.cpp)
target_link_libraries(bag2img ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(bag2pcd src/bag2pcd.cpp)
target_link_libraries(bag2pcd ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(localba src/localba.cpp)
target_link_libraries(localba ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(vis src/visualize.cpp)
target_link_libraries(vis ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(comb_imu src/comb_imu.cpp)
target_link_libraries(comb_imu ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(detect_aruco src/detect_aruco.cpp)
target_link_libraries(detect_aruco ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} gtsam)