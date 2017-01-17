cmake_minimum_required(VERSION 3.1)
project(hagrid)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

find_package(CUDA)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 11)
include_directories(lib/cub)
add_subdirectory(src)
