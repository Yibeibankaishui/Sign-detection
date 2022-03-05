cmake_minimum_required(VERSION 2.8)
project(cv_test)

set(CMAKE_BUILD_TYPE "Release")
# 添加c++ 11标准支持
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")


# 寻找OpenCV库
find_package(OpenCV REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(find_circle find_circle.cpp)
target_link_libraries(find_circle ${OpenCV_LIBS})

add_executable(sign_detect sign_detect.cpp)
target_link_libraries(sign_detect ${OpenCV_LIBS})

