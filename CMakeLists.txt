cmake_minimum_required(VERSION 3.1)
project(hagrid)

find_package(CUDA)
set(CMAKE_CXX_STANDARD 11)
include_directories(lib/cub)
add_subdirectory(src)
