cmake_minimum_required(VERSION 3.9)
project(dirt_detection VERSION 0.9.1 DESCRIPTION "Dirt detection on a dataset with floor mask.")



find_package( OpenCV 3.2 REQUIRED )
find_package(Boost COMPONENTS program_options filesystem REQUIRED)


include_directories(include ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})


add_executable(dirt_detection main.cpp visual_dirt_detection.cpp)
target_link_libraries(dirt_detection
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        )
