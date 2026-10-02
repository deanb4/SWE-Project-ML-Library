cmake_minimum_required(VERSION 3.18)
project(mllib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE MLLIB_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(mllib ${MLLIB_SOURCES})
target_include_directories(mllib PUBLIC include)
