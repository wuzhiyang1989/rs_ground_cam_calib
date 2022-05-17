cmake_minimum_required(VERSION 2.8)

project(camera_models)

find_package(catkin REQUIRED COMPONENTS cmake_modules)
include_directories(${catkin_INCLUDE_DIRS})

catkin_package(
    DEPENDS eigen opencv
    INCLUDE_DIRS include ${Eigen_INCLUDE_DIRS}
    LIBRARIES ${PROJECT_NAME}
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# FIND Gtest
enable_testing()
find_package(GTest REQUIRED)
if(GTest_FOUND)
    message(GTest found)
    message(GTEST_INCLUDE_DIR = ${GTEST_INCLUDE_DIR})
    message(GTEST_LIBRARIES = ${GTEST_LIBRARIES})
    include_directories(${GTEST_INCLUDE_DIR})
    #link_directories(${GTEST_LIBRARY_DIRS})
    add_definitions(${GTEST_DEFINITIONS})
else()
    message(err: CERES not found)
endif()

add_definitions(-fPIC -O3)
message(Eigen_INCLUDE_DIRS = ${EIGEN3_INCLUDE_DIR})
include_directories(include  ${EIGEN3_INCLUDE_DIR} ${catkin_INCLUDE_DIRS})

#library
file(GLOB SOURCE_FILES "src/*.cc")
message(SOURCE_FILES = ${SOURCE_FILES})
add_library(${PROJECT_NAME}  ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${GTEST_LIBRARIES} -lpthread ${catkin_LIBRARIES})