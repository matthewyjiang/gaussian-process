cmake_minimum_required(VERSION 3.17)
project(gaussian-process VERSION 0.1.0 LANGUAGES CXX)

# C++11 standard
set(CMAKE_CXX_STANDARD 11)


# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

include(FetchContent)

# Get Eigen 3.3.7
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://github.com/eigenteam/eigen-git-mirror
  GIT_TAG        cf794d3b741a6278df169e58461f8529f43bce5d
)
FetchContent_MakeAvailable(eigen)

# Create library
add_library(gp_lib STATIC ${SOURCES} ${HEADERS})
target_include_directories(gp_lib PUBLIC include ${eigen_SOURCE_DIR})

# Define test files and their corresponding executables
set(TESTS
    "test_gp:gp_test"
    "test_sine_rbf:test_sine_rbf"
    "test_quadratic_rbf:test_quadratic_rbf"
    "test_exponential_rbf:test_exponential_rbf"
    "test_logarithmic_rbf:test_logarithmic_rbf"
    "test_gaussian_rbf:test_gaussian_rbf"
)

# Create test executables
foreach(TEST_PAIR ${TESTS})
    string(REPLACE ":" ";" TEST_LIST ${TEST_PAIR})
    list(GET TEST_LIST 0 TEST_FILE)
    list(GET TEST_LIST 1 TEST_EXECUTABLE)
    
    add_executable(${TEST_EXECUTABLE} test/${TEST_FILE}.cpp)
    target_link_libraries(${TEST_EXECUTABLE} gp_lib)
endforeach()
