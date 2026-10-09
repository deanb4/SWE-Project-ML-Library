cmake_minimum_required(VERSION 3.18)
project(mllib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE MLLIB_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(mllib ${MLLIB_SOURCES})
target_include_directories(mllib PUBLIC include)

# Tests (GoogleTest is downloaded automatically at configure time)
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # use the same MSVC runtime as mllib
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(test_function tests/test_function.cpp)
target_link_libraries(test_function PRIVATE mllib GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_function)
