cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(bpvoTest)

set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/)

# Set the compiler options
#include(cmake/SetCompilerOptions.cmake)

set(CMAKE_CXX_FLAGS "-std=c++11")

add_executable(bpTest src/bpDescTest.cpp)
add_executable(bpVO src/bpVOTest.cpp)


find_package(OpenCV REQUIRED)

include_directories(/home/sourav/workspace/bpvo /home/sourav/workspace/bpvo/utils/)
include_directories(${OpenCV_INCLUDE_DIRS} /home/sourav/workspace/bpvo/eigen-eigen-07105f7124f9/)

target_link_libraries(bpTest ${OpenCV_LIBS} /home/sourav/workspace/bpvo/build/bin/libbpvo.a /home/sourav/workspace/bpvo/build/bin/libbpvo_utils.a)
target_link_libraries(bpVO ${OpenCV_LIBS} /home/sourav/workspace/bpvo/build/bin/libbpvo.a /home/sourav/workspace/bpvo/build/bin/libbpvo_utils.a)
