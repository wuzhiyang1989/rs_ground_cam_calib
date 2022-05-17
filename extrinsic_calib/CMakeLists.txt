cmake_minimum_required(VERSION 3.5.1)
project(extrinsic_calib)

# use c11 version
add_compile_options(-std=c++11)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

set(SRC_LIST_CALIB src/calib_main.cpp)

find_package(catkin REQUIRED COMPONENTS camera_models yaml-cpp)
include_directories(${catkin_INCLUDE_DIRS})
message(catkin_INCLUDE_DIRS = ${catkin_INCLUDE_DIRS})
message(catkin_LIBRARIES = ${catkin_LIBRARIES})

find_package(yaml-cpp)
message(yaml-cpp_INCLUDE_DIRS = ${yaml-cpp_INCLUDE_DIRS})
message(yaml-cpp_LIBRARIES = ${yaml-cpp_LIBRARIES})
include_directories(${yaml-cpp_INCLUDE_DIRS})

# find_package(Python2 COMPONENTS Interpreter Development REQUIRED)
# include_directories(${Python3_INCLUDE_DIRS})

catkin_python_setup()

catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  DEPENDS
)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    #link_directories(${OpenCV_LIBS_DIRS})
else()
    message(err: OpenCV not found)
endif()

find_package(Boost REQUIRED COMPONENTS serialization system filesystem program_options thread)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
    message(Boost_LIBRARY_DIRS = ${Boost_LIBRARY_DIRS})
    message(Boost_THREAD_LIBRARY = ${Boost_THREAD_LIBRARY})
    add_definitions(-DBOOST_ALL_DYN_LINK)
else()
    message(err: Boost not found)
endif()

find_package(Eigen3)
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    message(err: Eigen3 not found)
endif()

# FIND Ceres
find_package(Ceres REQUIRED)
if(CERES_FOUND)
    message(CERES found)
    message(CERES_INCLUDE_DIR = ${CERES_INCLUDE_DIR})
    message(CERES_LIBRARIES = ${CERES_LIBRARIES})
    include_directories(${CERES_INCLUDE_DIR})
    #link_directories(${CERES_LIBRARY_DIRS})
    add_definitions(${CERES_DEFINITIONS})
else()
    message(err: CERES not found)
endif()

add_executable(groundcam_extrinsic_calib ${SRC_LIST_CALIB})

target_link_libraries(groundcam_extrinsic_calib
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    ${CERES_LIBRARIES}
    ${catkin_LIBRARIES}
    ${Python3_LIBRARIES}
)
