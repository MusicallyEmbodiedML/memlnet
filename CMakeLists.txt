cmake_minimum_required(VERSION 3.29)
project(memlnet)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(libnpy/include)
add_executable(memlnet main_ch14.cpp libnpy/include/npy.hpp)
