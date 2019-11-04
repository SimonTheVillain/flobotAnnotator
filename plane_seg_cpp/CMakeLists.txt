cmake_minimum_required(VERSION 3.9)
project(plane_seg VERSION 0.9.1 DESCRIPTION "plane segmentation for pointclouds")

find_package(catkin REQUIRED COMPONENTS cv_bridge rosbag roscpp sensor_msgs tf pcl_ros)

find_package( OpenCV 3.2 REQUIRED )

find_package(PCL REQUIRED)

include_directories(include ${Boost_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})


add_executable(plane_seg main.cpp plane_segmentation_algorithm.cpp plane_segmentation_input.cpp)
target_link_libraries(plane_seg
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${rosbag_LIBRARIES}
        ${catkin_LIBRARIES}
        ${pcl_LIBRARIES}
        )
