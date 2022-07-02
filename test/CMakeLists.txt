cmake_minimum_required(VERSION 3.13)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.0.zip
)
FetchContent_MakeAvailable(googletest)

file(GLOB testSRC
    "*.cpp"
)
set(TEST_RUNNER main.cpp ${testSRC})
add_executable(tests_main ${TEST_RUNNER})

target_link_libraries(tests_main gtest llatvfl)
install(TARGETS tests_main DESTINATION bin)