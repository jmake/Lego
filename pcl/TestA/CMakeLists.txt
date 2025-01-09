cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(pcl-interactive_icp)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (test_a test_a.cpp)
target_link_libraries (test_a ${PCL_LIBRARIES})

## apt-get install -y cmake cmake-curses-gui build-essential
## apt-get install -y libpcl-dev
## apt-get install -y libflann-dev
