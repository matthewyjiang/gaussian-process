cmake_minimum_required(VERSION 3.17)
project(gaussian-process VERSION 0.1.0 LANGUAGES CXX)

# C++11 standard
set(CMAKE_CXX_STANDARD 11)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

# Create library
add_library(gp_lib STATIC ${SOURCES} ${HEADERS})
target_include_directories(gp_lib PUBLIC include)
target_link_libraries(gp_lib PUBLIC Eigen3::Eigen)

# Test executable
add_executable(gp_test test/test_gp.cpp)
target_link_libraries(gp_test gp_lib)

# Sine RBF test executable
add_executable(test_sine_rbf test/test_sine_rbf.cpp)
target_link_libraries(test_sine_rbf gp_lib)
