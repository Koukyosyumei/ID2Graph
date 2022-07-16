cmake_minimum_required(VERSION 3.13)

set( CMAKE_C_COMPILER mpicc )
set( CMAKE_CXX_COMPILER mpic++ )

add_executable(train_mpisecureboost pipeline_1_train_mpisecureboost.cpp)
add_executable(test_mpisecureboost test_mpisecureboost.cpp)
target_link_libraries(train_mpisecureboost boost_random boost_mpi boost_serialization pthread llatvfl)
target_link_libraries(test_mpisecureboost boost_random boost_mpi boost_serialization pthread llatvfl)