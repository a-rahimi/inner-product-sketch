cmake_minimum_required(VERSION 3.10)

project(MatrixProductSketch)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

find_package(GTest REQUIRED)

# Create tests
add_executable(test_matrix_product_sketch test_matrix_product_sketch.cpp)
target_link_libraries(test_matrix_product_sketch   GTest::gtest_main)
target_include_directories(test_matrix_product_sketch PUBLIC ${GTEST_INCLUDE_DIRS})

include(GoogleTest)
gtest_discover_tests(test_matrix_product_sketch)