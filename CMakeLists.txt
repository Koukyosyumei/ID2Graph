cmake_minimum_required(VERSION 3.13)
project("libllatvfl" LANGUAGES C CXX)

find_package(Threads REQUIRED)

set(CMAKE_EXECUTABLE_SUFFIX ".exe")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native -march=native")

set(SOURCE_DIR  "src/llatvfl")
set(TEST_DIR  "test")

include(GNUInstallDirs)

include_directories("${PROJECT_SOURCE_DIR}/src")

add_subdirectory(${SOURCE_DIR})

enable_testing()
add_subdirectory(${TEST_DIR})