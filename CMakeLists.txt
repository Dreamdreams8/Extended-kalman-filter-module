project(ExtendedKF)

cmake_minimum_required (VERSION 3.5)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV})

#set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/opencv3.4.19_installed/share/OpenCV)
# find_package(OpenCV 3.4.19 REQUIRED)  

find_package (Eigen3 REQUIRED)
include_directories(/usr/local/include/opencv4)
add_definitions(-std=c++0x)

set(sources
    src/extended_kalman_filter.cpp
    src/extended_kalman_module.cpp
    src/kalman_filter.cpp
    src/main.cpp
    src/tools.cpp)

add_executable(ExtendedKF ${sources})

# 链接OpenCV库
target_link_libraries(ExtendedKF ${OpenCV_LIBS})
