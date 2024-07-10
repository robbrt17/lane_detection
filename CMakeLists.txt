cmake_minimum_required(VERSION 3.10)
project(licenta)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCES src/Main.cpp includes/Utils.hpp src/Utils.cpp)
add_executable(licenta ${SOURCES})

target_include_directories(licenta PRIVATE src/ includes/)

target_link_libraries(licenta ${OpenCV_LIBS})
