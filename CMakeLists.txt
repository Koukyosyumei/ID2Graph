cmake_minimum_required(VERSION 3.13)
project("libllatvfl" LANGUAGES C CXX)

set(CMAKE_EXECUTABLE_SUFFIX ".exe")
set(SOURCE_DIR  "src/llatvfl")
set(TEST_DIR  "test")

include(GNUInstallDirs)

include_directories("${PROJECT_SOURCE_DIR}/src")

add_subdirectory(${SOURCE_DIR})
add_subdirectory(${TEST_DIR})