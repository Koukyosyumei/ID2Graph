cmake_minimum_required(VERSION 3.13)

#find_package(OpenMP REQUIRED)
#if(OpenMP_FOUND)
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

add_executable(train_randomforest pipeline_1_train_randomforest.cpp)
add_executable(train_secureboost pipeline_1_train_secureboost.cpp)
add_executable(train_xgboost pipeline_1_train_xgboost.cpp)
add_executable(demo_kmeans demo_kmeans.cpp)
add_executable(demo_randomforest_backdoor demo_randomforest_backdoor)

target_link_libraries(train_randomforest boost_random pthread llatvfl)
target_link_libraries(train_secureboost boost_random pthread llatvfl)
target_link_libraries(train_xgboost boost_random pthread llatvfl)
target_link_libraries(demo_kmeans pthread llatvfl)
target_link_libraries(demo_randomforest_backdoor boost_random pthread llatvfl)