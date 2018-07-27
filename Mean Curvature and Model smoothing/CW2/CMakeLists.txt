cmake_minimum_required(VERSION 3.9)
project(CW2)
include_directories(./spectra-master/include)

add_executable(${PROJECT_NAME}_bin main.cpp)
target_link_libraries(${PROJECT_NAME}_bin igl::core igl::viewer tutorials)