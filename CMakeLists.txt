cmake_minimum_required(VERSION 3.17)
project(gaussian-process VERSION 0.2.0 LANGUAGES CXX)

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

# Define test files and their corresponding executables
set(TESTS
    "test_gp:gp_test"
    "test_sine_rbf:test_sine_rbf"
    "test_quadratic_rbf:test_quadratic_rbf"
    "test_exponential_rbf:test_exponential_rbf"
    "test_logarithmic_rbf:test_logarithmic_rbf"
    "test_gaussian_rbf:test_gaussian_rbf"
    "test_add_data:test_add_data"
    "test_incremental_learning:test_incremental_learning"
)


# Create test executables
foreach(TEST_PAIR ${TESTS})
    string(REPLACE ":" ";" TEST_LIST ${TEST_PAIR})
    list(GET TEST_LIST 0 TEST_FILE)
    list(GET TEST_LIST 1 TEST_EXECUTABLE)
    
    add_executable(${TEST_EXECUTABLE} test/${TEST_FILE}.cpp)
    target_link_libraries(${TEST_EXECUTABLE} gp_lib)
endforeach()
